"""İki VLM'i aynı fotoğraflarda yan yana ölçer — öznitelik çıkarım kalitesi.

Neden ayrı betik: `scripts/vlm_toplu.py` çıktıyı her zaman
`data/oznitelikler.jsonl`'e yazıyor. Farklı bir modeli onunla denemek, mevcut
taban çizgisini üzerine yazmak demek. Karşılaştırma yapabilmek için ikisinin de
aynı anda durması şart, o yüzden burada çıktı yolu parametre.

Ölçülen asıl şey MARKA. 11 Ağustos ölçümü VLM'in markayı ürünlerin ancak
%43'ünde okuyabildiğini gösterdi ve marka hem İndeks B'nin cümlesine hem İndeks
C'nin FTS'ine gidiyor — yani tek bir iyileşme iki indeksi birden besliyor.
Kategori zaten %100, oradan kazanç beklenmiyor.

Dikkat: "daha çok alan doldu" tek başına iyi haber DEĞİL. Büyük model markayı
uydurmaya da başlayabilir. Bu yüzden karşılaştırma, iki modelin AYNI fotoğrafta
ne dediğini yan yana basıyor; sayı tek başına karar verdirmiyor.

    .venv\\Scripts\\python.exe eval\\kiyas_vlm.py calistir Qwen/Qwen3-VL-8B-Instruct data\\oznitelikler_8b.jsonl
    .venv\\Scripts\\python.exe eval\\kiyas_vlm.py karsilastir data\\oznitelikler.jsonl data\\oznitelikler_8b.jsonl
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

ALANLAR = ("kategori", "marka", "renk", "malzeme", "durum", "ayirt_edici")
BOS_DEGERLER = {"", "bilinmiyor", "bilinmeyen", "yok", "none", "null"}

# vlm_toplu.py ile birebir aynı olmalı, yoksa kıyas bozulur: aynı prompt, aynı
# çözünürlük, aynı üretim ayarları. Tek değişken model olsun.
VARSAYILAN_PROMPT = "oznitelik_renkli.txt"
UZUN_KENAR = 896
AZAMI_TOKEN = 200


def dolu(deger) -> bool:
    return deger is not None and str(deger).strip().lower() not in BOS_DEGERLER


def json_ayikla(ham: str) -> dict | None:
    """vlm_toplu.py'deki ile aynı ayrıştırma — model bazen ```json çitiyle sarıyor."""
    temiz = re.sub(r"^```(?:json)?\s*", "", ham.strip())
    temiz = re.sub(r"\s*```$", "", temiz)
    try:
        return json.loads(temiz)
    except json.JSONDecodeError:
        pass
    esles = re.search(r"\{.*\}", temiz, re.DOTALL)
    if esles:
        try:
            return json.loads(esles.group(0))
        except json.JSONDecodeError:
            return None
    return None


def calistir(model_id: str, cikti_yolu: Path, prompt_adi: str) -> int:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if cikti_yolu.exists():
        print(f"[!] {cikti_yolu} zaten var. Üzerine yazmamak için önce taşı veya sil.")
        return 1

    prompt = (config.PROMPT_DIZINI / prompt_adi).read_text(encoding="utf-8")
    fotograflar = sorted(
        p for p in config.FOTO_DIZINI.rglob("*")
        if p.suffix.lower() in config.RESIM_UZANTILARI
    )
    if not fotograflar:
        print(f"[!] {config.FOTO_DIZINI} içinde fotoğraf yok")
        return 1

    # lm_head kararı — 18 Ağustos ölçümünden çıkan zorunluluk.
    #
    # bitsandbytes lm_head'i varsayılan olarak quantize ETMEZ. 4B'de bu sorun
    # değil çünkü tie_word_embeddings=true, yani ayrı bir lm_head matrisi yok.
    # 8B'de tie yok: lm_head 622M parametre ve bf16'da 1.16 GB yer tutuyor.
    # Bu 1.16 GB, 8 GB kartta tepe VRAM'i 6.81 GB'a çıkarıyor ve kart tavana
    # dayanınca Windows'un WDDM sürücüsü OOM ATMAK YERİNE GPU belleğini sessizce
    # sistem RAM'ine sayfalıyor. Sonuç: hata yok, exit 0, ama fotoğraf başına
    # süre 6 saniyeden 4669 saniyeye çıkıyor (~780x). Sessiz olduğu için en
    # tehlikeli hata biçimi bu.
    #
    # lm_head'i de NF4'e sokmak tepeyi 6.07 GB'a indiriyor ve süre 7.7 saniyeye
    # dönüyor. Bedeli var: lm_head doğrudan logit üretiyor, quantize etmek
    # üretim kalitesine dokunuyor. Bu yüzden yalnızca GEREKTİĞİNDE yapılıyor —
    # tie varsa hiç dokunulmuyor.
    # Bayrağın YERİ modele göre değişiyor ve bu bir tuzak:
    #   4B -> üst düzeyde True,  text_config'de de True
    #   8B -> üst düzeyde False, text_config'de HİÇ YOK
    # Yani text_config'e bakıp varsayılana düşmek 8B'yi "tie var" sanmaya
    # yetiyor. İlk yazımda tam bu oldu ve koşu sayfalama yoluna girdi.
    #
    # Karar bilerek asimetrik: iki hatanın bedeli eşit değil.
    #   gereksiz quantize -> zararsız, tie varsa ayrı lm_head yok, işlem no-op
    #   gerekirken atlama -> fotoğraf başına 6 sn yerine 4669 sn, sessizce
    # O yüzden varsayılan "quantize et"; yalnızca tie KESİN doğrulanırsa atla.
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_id)
    tie = None
    for kaynak in (model_config, getattr(model_config, "text_config", None)):
        if kaynak is not None and hasattr(kaynak, "tie_word_embeddings"):
            tie = bool(getattr(kaynak, "tie_word_embeddings"))
            break
    atlanacak = None if tie is True else []

    nicemleme = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=atlanacak,
    )
    print(f"[i] {len(fotograflar)} fotoğraf · model: {model_id} · prompt: {prompt_adi}")
    print(f"[i] tie_word_embeddings={tie} -> lm_head "
          f"{'bf16 bırakılıyor (tie var, ayrı matris yok)' if tie is True else 'DA NF4 ediliyor (sayfalamayı önlemek için)'}")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    islemci = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=nicemleme, device_map={"": 0}, dtype=torch.bfloat16,
    ).eval()
    yukleme = time.perf_counter() - t0
    agirlik = torch.cuda.memory_allocated() / 1024**3
    print(f"[+] Yüklendi ({yukleme:.0f} sn) · ağırlık VRAM: {agirlik:.2f} GB\n", flush=True)

    kayitlar: list[dict] = []
    basarisiz = 0
    t_baslangic = time.perf_counter()

    for i, foto in enumerate(fotograflar, 1):
        gorsel = Image.open(foto).convert("RGB")
        gorsel.thumbnail((UZUN_KENAR, UZUN_KENAR), Image.Resampling.LANCZOS)
        mesaj = [{"role": "user", "content": [
            {"type": "image", "image": gorsel}, {"type": "text", "text": prompt}]}]
        girdi = islemci.apply_chat_template(
            mesaj, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            cikti = model.generate(**girdi, max_new_tokens=AZAMI_TOKEN, do_sample=False)
        ham = islemci.decode(
            cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True)

        veri = json_ayikla(ham)
        if veri is None:
            basarisiz += 1
            print(f"  [{i}/{len(fotograflar)}] {foto.name}  [!] JSON ayrıştırılamadı")
            continue
        veri["dosya"] = foto.name
        kayitlar.append(veri)

        if i % 10 == 0 or i == len(fotograflar):
            gecen = time.perf_counter() - t_baslangic
            tepe = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [{i}/{len(fotograflar)}] {gecen / i:.1f} sn/fotoğraf · "
                  f"tepe VRAM {tepe:.2f} GB", flush=True)

    toplam = time.perf_counter() - t_baslangic
    cikti_yolu.write_text(
        "\n".join(json.dumps(k, ensure_ascii=False) for k in kayitlar) + "\n",
        encoding="utf-8")
    print(f"\n[+] {len(kayitlar)} kayıt -> {cikti_yolu}")
    print(f"[i] JSON başarısız: {basarisiz}/{len(fotograflar)}")
    print(f"[i] Tepe VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB / 8.00 GB")
    print(f"[i] Süre: yükleme {yukleme:.0f} sn + işleme {toplam:.0f} sn "
          f"({toplam / max(len(fotograflar), 1):.1f} sn/fotoğraf)")
    return 0


def yukle(yol: Path) -> dict[str, dict]:
    """dosya adı -> öznitelik. Kayıtlar fotoğrafa göre eşleşsin diye sözlük."""
    return {
        k["dosya"]: k
        for k in (json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip())
    }


def karsilastir(taban_yolu: Path, yeni_yolu: Path) -> int:
    taban, yeni = yukle(taban_yolu), yukle(yeni_yolu)
    ortak = sorted(set(taban) & set(yeni))
    if not ortak:
        print("[!] İki dosyada ortak fotoğraf yok.")
        return 1
    print(f"[i] {len(ortak)} ortak fotoğraf  ({taban_yolu.name} -> {yeni_yolu.name})\n")

    print("=" * 70)
    print("ALAN DOLULUĞU")
    print("=" * 70)
    print(f"  {'alan':<14}{'taban':>10}{'yeni':>10}{'fark':>10}")
    print("  " + "-" * 44)
    for alan in ALANLAR:
        t = sum(1 for d in ortak if dolu(taban[d].get(alan)))
        y = sum(1 for d in ortak if dolu(yeni[d].get(alan)))
        isaret = f"{y - t:+d}" if y != t else "—"
        print(f"  {alan:<14}{t:>7}/{len(ortak):<3}{y:>7}/{len(ortak):<3}{isaret:>10}")

    print(f"\n{'=' * 70}")
    print("ANLAŞMA — aynı fotoğrafta aynı değeri mi söylüyorlar?")
    print("=" * 70)
    for alan in ALANLAR:
        ikisi_dolu = [d for d in ortak if dolu(taban[d].get(alan)) and dolu(yeni[d].get(alan))]
        if not ikisi_dolu:
            continue
        ayni = sum(1 for d in ikisi_dolu
                   if str(taban[d][alan]).strip().lower() == str(yeni[d][alan]).strip().lower())
        print(f"  {alan:<14}{ayni}/{len(ikisi_dolu)} aynı ({ayni / len(ikisi_dolu) * 100:.0f}%)")

    # Marka asıl mesele: yeni modelin okuduğu, tabanın okuyamadığı markalar.
    # Bunlar tek tek gözle doğrulanmalı — uydurma mı, gerçekten okumuş mu.
    print(f"\n{'=' * 70}")
    print("MARKA — yeni modelde DOLAN kayıtlar (gözle doğrula: uydurma mı?)")
    print("=" * 70)
    dolan = [d for d in ortak if not dolu(taban[d].get("marka")) and dolu(yeni[d].get("marka"))]
    for d in dolan:
        print(f"  + {d:<16} {yeni[d]['marka']}")
    print(f"  toplam: {len(dolan)}")

    kaybolan = [d for d in ortak if dolu(taban[d].get("marka")) and not dolu(yeni[d].get("marka"))]
    print(f"\n  Yeni modelde KAYBOLAN marka: {len(kaybolan)}")
    for d in kaybolan:
        print(f"  - {d:<16} taban: {taban[d]['marka']}")

    celisen = [d for d in ortak
               if dolu(taban[d].get("marka")) and dolu(yeni[d].get("marka"))
               and str(taban[d]["marka"]).strip().lower() != str(yeni[d]["marka"]).strip().lower()]
    print(f"\n  ÇELİŞEN marka: {len(celisen)}")
    for d in celisen:
        print(f"  ! {d:<16} taban: {taban[d]['marka']:<20} yeni: {yeni[d]['marka']}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    komut = sys.argv[1]
    if komut == "calistir":
        if len(sys.argv) < 4:
            print("Kullanım: kiyas_vlm.py calistir <model_id> <cikti.jsonl> [prompt]")
            return 1
        prompt_adi = sys.argv[4] if len(sys.argv) > 4 else VARSAYILAN_PROMPT
        return calistir(sys.argv[2], Path(sys.argv[3]), prompt_adi)
    if komut == "karsilastir":
        if len(sys.argv) < 4:
            print("Kullanım: kiyas_vlm.py karsilastir <taban.jsonl> <yeni.jsonl>")
            return 1
        return karsilastir(Path(sys.argv[2]), Path(sys.argv[3]))
    print(f"[!] Bilinmeyen komut: {komut}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
