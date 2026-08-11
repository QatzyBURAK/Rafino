"""VLM izole testi — Qwen3-VL-4B-Instruct, 4-bit.

Modeli tek başına yükler, tek fotoğrafta çalıştırır, ölçülen VRAM'i ve ham
çıktıyı basar. Kayıt hattına bağlamadan önce şunları görmek için:

  - 4-bit nicemleme gerçekten çalışıyor mu, ne kadar VRAM tutuyor
  - prompt JSON döndürüyor mu, yoksa ```json ile mi sarıyor
  - kategori kuralı tutuyor mu ("gumruk" gibi kelimeler sızıyor mu)
  - renk yasağına uyuyor mu

Kullanım:
    .venv\\Scripts\\python.exe scripts\\test_vlm.py <fotograf yolu> [baska.jpg ...]

Bu betik AYRI BİR SÜREÇ olarak çalıştırılmalı. Süreç bitince işletim sistemi
VRAM'i eksiksiz geri alır; aynı süreç içinde `del` + `empty_cache` bunu garanti
etmiyor.
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402


def vram_gb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return (0.0, 0.0)
    bos, toplam = torch.cuda.mem_get_info()
    return ((toplam - bos) / 1024**3, toplam / 1024**3)


def json_ayikla(ham: str) -> dict | None:
    """Modelin çıktısından JSON'u güvenle çıkarır.

    Model ```json ile sarabiliyor, önüne açıklama ekleyebiliyor. Doküman
    bunu 'güvenli parse fonksiyonu şart' diye not etmiş; burası o fonksiyon.
    """
    temiz = ham.strip()
    temiz = re.sub(r"^```(?:json)?\s*", "", temiz)
    temiz = re.sub(r"\s*```$", "", temiz)
    try:
        return json.loads(temiz)
    except json.JSONDecodeError:
        pass
    # Metnin içine gömülmüşse ilk süslü parantez bloğunu dene.
    esles = re.search(r"\{.*\}", temiz, re.DOTALL)
    if esles:
        try:
            return json.loads(esles.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    yollar = [Path(a) for a in sys.argv[1:]]
    if not yollar:
        print(__doc__)
        return 1
    eksik = [p for p in yollar if not p.exists()]
    if eksik:
        for p in eksik:
            print(f"[!] Dosya yok: {p}")
        return 1

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    prompt = (config.PROMPT_DIZINI / "oznitelik.txt").read_text(encoding="utf-8")
    print(f"[i] Prompt okundu: {len(prompt)} karakter")

    kullanim_once, toplam = vram_gb()
    print(f"[i] Yükleme öncesi VRAM: {kullanim_once:.2f} / {toplam:.1f} GB")

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
        # device_map="auto" 4-bit ile bazı katmanları CPU'ya atıp patlıyor.
        # Tek GPU olduğu için doğrudan 0'a sabitliyoruz.
        device_map={"": 0},
        dtype=torch.bfloat16,
    )
    model.eval()
    yukleme_sn = time.perf_counter() - t0

    kullanim_sonra, _ = vram_gb()
    print(f"[+] Yüklendi. {yukleme_sn:.1f} sn")
    print(f"[+] VRAM: {kullanim_sonra:.2f} GB (model ~{kullanim_sonra - kullanim_once:.2f} GB)")

    from PIL import Image

    basarili = 0
    for yol in yollar:
        print(f"\n{'=' * 60}\n{yol.name}\n{'=' * 60}")
        gorsel = Image.open(yol).convert("RGB")
        mesajlar = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": gorsel},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        girdi = islemci.apply_chat_template(
            mesajlar,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.inference_mode():
            cikti = model.generate(**girdi, max_new_tokens=256, do_sample=False)
        uretim_sn = time.perf_counter() - t0

        yeni = cikti[0][girdi["input_ids"].shape[1] :]
        ham = islemci.decode(yeni, skip_special_tokens=True)

        print(f"[ham çıktı] ({uretim_sn:.1f} sn)")
        print(ham)

        veri = json_ayikla(ham)
        if veri is None:
            print("[!] JSON ayrıştırılamadı")
            continue
        basarili += 1
        print("\n[ayrıştırılmış]")
        for anahtar, deger in veri.items():
            print(f"    {anahtar:<12} : {deger}")

        # Kural ihlali kontrolleri
        metin = json.dumps(veri, ensure_ascii=False).lower()
        renkler = ["mavi", "siyah", "kirmizi", "kırmızı", "beyaz", "yesil", "yeşil",
                   "sari", "sarı", "gri", "pembe", "mor", "turuncu", "kahverengi"]
        sizan = [r for r in renkler if r in metin]
        if sizan:
            print(f"    [!] RENK SIZDI: {sizan}  <- prompt sıkılaştırılmalı")
        yasak = ["gumruk", "gümrük", "tasfiye", "depo", "esya", "eşya", "urun", "ürün"]
        sizan2 = [k for k in yasak if k in str(veri.get("kategori", "")).lower()]
        if sizan2:
            print(f"    [!] GENEL KELİME kategoriye sızdı: {sizan2}")

    zirve, _ = vram_gb()
    print(f"\n{'=' * 60}")
    print(f"[i] Zirve VRAM: {zirve:.2f} / {toplam:.1f} GB")
    print(f"[i] JSON başarı: {basarili}/{len(yollar)}")
    return 0 if basarili == len(yollar) else 1


if __name__ == "__main__":
    raise SystemExit(main())
