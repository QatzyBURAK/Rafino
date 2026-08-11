"""Değerlendirme setini kurar: fotoğraflar + ground truth + Türkçe sorgular.

Kaynak: PestoRosso/lamoda-fashion-product-images (MIT lisans, Kaggle Fashion
Product Images veri setinin yüksek çözünürlüklü alt kümesi). Görseller
1080x1440 ile 1800x2400 arasında, yani VLM etiketten marka okuyabiliyor.

Bu veri setinin seçilme sebebi görselleri değil ETİKETLERİ: her ürünün
kategorisi, rengi ve adı hazır geliyor. Böylece değerlendirme seti elle
etiketlenmiyor, ground truth'tan otomatik üretiliyor.

Örnekleme rastgele DEĞİL. Kasıtlı olarak aynı kategoride farklı renkli ürünler
seçiliyor, çünkü projenin asıl iddiası tam burada sınanıyor: "mavi el çantası"
sorgusu, siyah el çantasını ve mavi ayakkabıyı ELEYEBİLMELİ. Rastgele örneklemde
bu çiftler oluşmayabilir ve test kolaylaşır.

    .venv\\Scripts\\python.exe scripts\\veriseti_kur.py [urun_sayisi]
"""

import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from PIL import Image  # noqa: E402

from src import config  # noqa: E402

VERI_SETI = "PestoRosso/lamoda-fashion-product-images"
PARCA = "data/train_part_0000.parquet"
LISANS = "MIT"

# Depo bağlamına uyan kategoriler. Giyim ağırlığı bilerek düşük tutuldu:
# gümrük deposunda çanta, saat, gözlük, parfüm, cüzdan daha temsili.
#
# AYAKKABI KATEGORİLERİ BİLEREK 2'YE İNDİRİLDİ. İlk denemede Sandals, Heels,
# Casual Shoes ve Sports Shoes birlikte seçilmişti ve sonuçlar bozuldu. İki
# sebeple:
#   1) Kaynak veri setinin etiketleri bu alt türlerde tutarsız. "Sandals"
#      etiketli ürünler arasında Crocs terliği ve parmak arası terlik var
#      (üstelik ayrı bir "Flip Flops" kategorisi de mevcut); "Heels" etiketli
#      ürünlerin adları "Flats" ve "Wedges" diyor.
#   2) Dört ayakkabı alt türü birbirini karıştırıyor ve ölçüm, modelin
#      yeteneğini değil etiket gürültüsünü ölçüyor.
# Kalan iki tür (spor / günlük) görsel olarak yeterince ayrık.
KATEGORILER = {
    "Handbags": "el çantası",
    "Backpacks": "sırt çantası",
    "Watches": "saat",
    "Casual Shoes": "günlük ayakkabı",
    "Sports Shoes": "spor ayakkabı",
    "Belts": "kemer",
    "Wallets": "cüzdan",
    "Sunglasses": "güneş gözlüğü",
    "Perfume and Body Mist": "parfüm",
    "Caps": "şapka",
    "Tshirts": "tişört",
    "Shirts": "gömlek",
    "Jeans": "kot pantolon",
    "Trousers": "pantolon",
    "Lipstick": "ruj",
}

RENKLER = {
    "Black": "siyah",
    "White": "beyaz",
    "Blue": "mavi",
    "Navy Blue": "lacivert",
    "Brown": "kahverengi",
    "Grey": "gri",
    "Red": "kırmızı",
    "Pink": "pembe",
    "Green": "yeşil",
    "Silver": "gümüş",
    "Purple": "mor",
    "Beige": "bej",
    "Yellow": "sarı",
    "Maroon": "bordo",
    "Gold": "altın",
    "Orange": "turuncu",
    "Olive": "zeytin yeşili",
    "Charcoal": "antrasit",
    "Cream": "krem",
    "Copper": "bakır",
    "Off White": "kırık beyaz",
    "Steel": "çelik",
    "Teal": "petrol mavisi",
    "Tan": "ten rengi",
    "Lavender": "lavanta",
    "Coffee Brown": "kahve",
    "Rose": "gül kurusu",
    "Mauve": "leylak",
    "Turquoise Blue": "turkuaz",
    "Burgundy": "bordo",
    "Khaki": "haki",
    "Mustard": "hardal",
    "Peach": "şeftali",
    "Magenta": "fuşya",
    "Rust": "kiremit",
    "Bronze": "bronz",
    "Nude": "ten rengi",
    "Sea Green": "deniz yeşili",
    "Lime Green": "fıstık yeşili",
    "Grey Melange": "gri melanj",
    "Multi": "çok renkli",
}

# Kategori başına en fazla kaç ürün. Tek kategorinin seti ele geçirmesini önler.
KATEGORI_BASI_TAVAN = 6
# Bir kategori-renk çiftinden en fazla kaç ürün. Ayrım testi için 2 yeterli.
CIFT_BASI_TAVAN = 2


def marka_cikar(urun_adi: str) -> str:
    """Ürün adının ilk kelimesi genelde marka: 'Puma Men Grey T-shirt' -> Puma.

    Kusursuz değil ama İndeks C (anahtar kelime) testi için yeterli; marka
    ground truth'u zaten bu alandan üretiliyor, yani tutarlı.
    """
    return urun_adi.split()[0] if urun_adi else "bilinmiyor"


def main() -> int:
    hedef = int(sys.argv[1]) if len(sys.argv) > 1 else 60

    config.dizinleri_hazirla()
    print(f"[i] Kaynak: {VERI_SETI} ({LISANS})")
    yol = hf_hub_download(VERI_SETI, PARCA, repo_type="dataset")

    tablo = pq.read_table(
        yol,
        columns=["id", "article_type", "base_color", "product_display_name", "image"],
    )
    print(f"[i] Parçada {tablo.num_rows} ürün var, {hedef} tanesi seçilecek")

    idler = tablo.column("id").to_pylist()
    tipler = tablo.column("article_type").to_pylist()
    renkler = tablo.column("base_color").to_pylist()
    adlar = tablo.column("product_display_name").to_pylist()

    # Önce uygun adayları kategori-renk çiftine göre grupla.
    gruplar: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, (tip, renk) in enumerate(zip(tipler, renkler)):
        if tip in KATEGORILER and renk in RENKLER:
            gruplar[(tip, renk)].append(i)

    # Renk çeşitliliği yüksek kategorileri öne al: ayrım testi oradan çıkıyor.
    kategori_renk_sayisi: dict[str, int] = defaultdict(int)
    for tip, _ in gruplar:
        kategori_renk_sayisi[tip] += 1

    secilen: list[int] = []
    kategori_sayaci: dict[str, int] = defaultdict(int)

    # Tur tur ilerle: her turda her kategoriden bir renk al. Böylece hiçbir
    # kategori tek başına dolmadan diğerlerine sıra gelir.
    ciftler = sorted(
        gruplar.items(),
        key=lambda x: (-kategori_renk_sayisi[x[0][0]], x[0][0], -len(x[1])),
    )
    for tur in range(CIFT_BASI_TAVAN):
        for (tip, _renk), indeksler in ciftler:
            if len(secilen) >= hedef:
                break
            if kategori_sayaci[tip] >= KATEGORI_BASI_TAVAN:
                continue
            if tur < len(indeksler):
                secilen.append(indeksler[tur])
                kategori_sayaci[tip] += 1
        if len(secilen) >= hedef:
            break

    print(f"[i] {len(secilen)} ürün seçildi, {len(kategori_sayaci)} kategoriden")

    # --- Görselleri diske yaz, ground truth topla ---
    urunler = []
    for sira, i in enumerate(secilen, 1):
        tip_en, renk_en = tipler[i], renkler[i]
        kategori = KATEGORILER[tip_en]
        renk = RENKLER[renk_en]
        urun_adi = adlar[i]
        marka = marka_cikar(urun_adi)

        # Parquet'te görsel ya düz bayt ya da {"bytes": ..., "path": ...} olarak
        # duruyor; hangi dışa aktarımla üretildiğine bağlı. İkisini de karşıla.
        ham = tablo.column("image")[i].as_py()
        baytlar = ham["bytes"] if isinstance(ham, dict) else ham
        gorsel = Image.open(io.BytesIO(baytlar)).convert("RGB")

        dosya_adi = f"{idler[i]}.jpg"
        gorsel.save(config.FOTO_DIZINI / dosya_adi, "JPEG", quality=92)

        urunler.append({
            "id": idler[i],
            "dosya": dosya_adi,
            "kategori": kategori,
            "renk": renk,
            "marka": marka,
            "kategori_en": tip_en,
            "renk_en": renk_en,
            "urun_adi": urun_adi,
            # Raf numarası uydurma değil, sistemin alanı — demo için atanıyor.
            "raf": f"{chr(65 + sira % 8)}-{sira:02d}",
        })
        if sira % 10 == 0:
            print(f"    {sira}/{len(secilen)} yazıldı")

    (config.EVAL_DIZINI / "urunler.jsonl").write_text(
        "\n".join(json.dumps(u, ensure_ascii=False) for u in urunler) + "\n",
        encoding="utf-8",
    )
    print(f"[+] {len(urunler)} fotoğraf -> {config.FOTO_DIZINI}")

    # --- Sorguları üret ---
    # Üç zorluk kademesi, üç indekse denk geliyor.
    sorgular = []

    kategori_gruplari: dict[str, list[int]] = defaultdict(list)
    cift_gruplari: dict[tuple[str, str], list[int]] = defaultdict(list)
    marka_gruplari: dict[str, list[int]] = defaultdict(list)
    for u in urunler:
        kategori_gruplari[u["kategori"]].append(u["id"])
        cift_gruplari[(u["renk"], u["kategori"])].append(u["id"])
        marka_gruplari[u["marka"]].append(u["id"])

    # 1) Sadece kategori — kolay. Görsel indeks bunu zaten çözmeli.
    for kategori, idl in kategori_gruplari.items():
        sorgular.append({
            "sorgu": kategori,
            "dogru": sorted(idl),
            "tip": "kategori",
            "zorluk": "kolay",
        })

    # 2) Renk + kategori — ZOR. Projenin asıl iddiası bu.
    #    Aynı kategoride başka renk varsa gerçek ayrım testi olur.
    for (renk, kategori), idl in cift_gruplari.items():
        ayni_kategori_baska_renk = len(kategori_gruplari[kategori]) > len(idl)
        sorgular.append({
            "sorgu": f"{renk} {kategori}",
            "dogru": sorted(idl),
            "tip": "renk+kategori",
            "zorluk": "zor" if ayni_kategori_baska_renk else "orta",
        })

    # 3) Marka — anahtar kelime indeksinin işi (İndeks C).
    for marka, idl in marka_gruplari.items():
        if len(marka) > 2:
            sorgular.append({
                "sorgu": marka,
                "dogru": sorted(idl),
                "tip": "marka",
                "zorluk": "anahtar kelime",
            })

    (config.EVAL_DIZINI / "sorgular.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in sorgular) + "\n",
        encoding="utf-8",
    )

    # --- Özet ---
    from collections import Counter

    zorluk = Counter(s["zorluk"] for s in sorgular)
    print(f"[+] {len(sorgular)} sorgu -> {config.EVAL_DIZINI / 'sorgular.jsonl'}")
    for z, adet in zorluk.most_common():
        print(f"      {z:<16} {adet}")

    print("\n=== Kategori dağılımı ===")
    for kategori, idl in sorted(kategori_gruplari.items(), key=lambda x: -len(x[1])):
        renkler_str = ", ".join(sorted({u["renk"] for u in urunler if u["kategori"] == kategori}))
        print(f"  {kategori:<20} {len(idl):>2} ürün   ({renkler_str})")

    zor = [s for s in sorgular if s["zorluk"] == "zor"]
    print(f"\n=== Ayrım testi sorguları ({len(zor)} tane) ===")
    for s in zor[:12]:
        print(f"  \"{s['sorgu']}\"  -> {len(s['dogru'])} doğru cevap")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
