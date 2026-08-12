"""VLM'i tüm fotoğraflarda TOPLU çalıştırır ve öznitelik JSON'larını kaydeder.

Neden toplu (11 Ağustos ölçümünden çıkan karar): VLM yüklemesi 69-110 saniye
sürüyor ama bir fotoğrafı 4-5 saniyede işliyor. Fotoğraf başına ayrı süreç
açmak, yükleme bedelini her seferinde ödemek demek. Tek süreçte 60 fotoğraf
işlenince bu bedel 60'a bölünüyor.

Süreç bittiğinde işletim sistemi VRAM'i eksiksiz geri alır; bu yüzden bu betik
ayrı bir süreç olarak çalıştırılıyor ve sonuçlar diske yazılıyor. Kayıt hattının
geri kalanı bu dosyayı okur, VLM'i bir daha yüklemez.

    .venv\\Scripts\\python.exe scripts\\vlm_toplu.py [prompt_dosyasi]
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

CIKTI = config.VERI_DIZINI / "oznitelikler.jsonl"


def json_ayikla(ham: str) -> dict | None:
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


def main() -> int:
    prompt_adi = sys.argv[1] if len(sys.argv) > 1 else "oznitelik_renkli.txt"
    prompt = (config.PROMPT_DIZINI / prompt_adi).read_text(encoding="utf-8")

    fotograflar = sorted(
        p for p in config.FOTO_DIZINI.rglob("*")
        if p.suffix.lower() in config.RESIM_UZANTILARI
    )
    if not fotograflar:
        print(f"[!] {config.FOTO_DIZINI} içinde fotoğraf yok")
        return 1

    print(f"[i] {len(fotograflar)} fotoğraf, prompt: {prompt_adi}")

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    nicemleme = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[i] Yükleniyor: {config.VLM_MODEL} (4-bit NF4)")
    t0 = time.perf_counter()
    islemci = AutoProcessor.from_pretrained(config.VLM_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        config.VLM_MODEL,
        quantization_config=nicemleme,
        device_map={"": 0},
        dtype=torch.bfloat16,
    ).eval()
    yukleme = time.perf_counter() - t0
    print(f"[+] Yüklendi ({yukleme:.0f} sn). Fotoğraflara geçiliyor.\n")

    kayitlar: list[dict] = []
    basarisiz = 0
    t_baslangic = time.perf_counter()

    for i, foto in enumerate(fotograflar, 1):
        gorsel = Image.open(foto).convert("RGB")
        # Qwen3-VL dinamik çözünürlük kullanıyor: görsel büyüdükçe görü token
        # sayısı, VRAM ve süre birlikte artıyor. 1800x2400 fotoğraflarda VRAM
        # 3.9 GB yerine 7.2 GB'a çıktı ve fotoğraf başına süre kat kat uzadı.
        # Öznitelik çıkarımı (kategori, marka, malzeme) için bu çözünürlük
        # gereksiz; uzun kenar 896 piksel etiket okumaya fazlasıyla yetiyor.
        gorsel.thumbnail((896, 896), Image.Resampling.LANCZOS)
        mesaj = [{
            "role": "user",
            "content": [
                {"type": "image", "image": gorsel},
                {"type": "text", "text": prompt},
            ],
        }]
        girdi = islemci.apply_chat_template(
            mesaj, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            cikti = model.generate(**girdi, max_new_tokens=200, do_sample=False)
        ham = islemci.decode(
            cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True
        )

        veri = json_ayikla(ham)
        if veri is None:
            basarisiz += 1
            print(f"  [{i}/{len(fotograflar)}] {foto.name}  [!] JSON ayrıştırılamadı")
            continue

        veri["dosya"] = foto.name
        kayitlar.append(veri)

        if i % 10 == 0 or i == len(fotograflar):
            gecen = time.perf_counter() - t_baslangic
            # flush şart: çıktı bir dosyaya yönlendirildiğinde Python stdout'u
            # tamponluyor ve ilerleme ancak süreç bitince görünüyor.
            print(f"  [{i}/{len(fotograflar)}] {gecen / i:.1f} sn/fotoğraf", flush=True)

    toplam = time.perf_counter() - t_baslangic
    CIKTI.write_text(
        "\n".join(json.dumps(k, ensure_ascii=False) for k in kayitlar) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 58}")
    print(f"[+] {len(kayitlar)} kayıt -> {CIKTI}")
    print(f"[i] JSON başarısız: {basarisiz}/{len(fotograflar)}")
    print(f"[i] Süre: yükleme {yukleme:.0f} sn + işleme {toplam:.0f} sn "
          f"({toplam / len(fotograflar):.1f} sn/fotoğraf)")
    print(f"[i] Fotoğraf başına ayrı süreç açılsaydı: "
          f"~{(yukleme + toplam / len(fotograflar)) * len(fotograflar) / 60:.0f} dakika "
          f"(şimdi {(yukleme + toplam) / 60:.0f} dakika)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
