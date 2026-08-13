"""Kopya tespiti eşiklerini ÖLÇEREK belirler.

Sorun: "aynı ürün" ile "benzer ürün"ü ayıran cosine eşiği kaç olmalı? Tahminle
konursa ya kopyalar kaçar ya farklı ürünler yanlışlıkla birleştirilir. İkincisi
daha kötü: stok kaydı sessizce bozulur.

Elimizde aynı ürünün ikinci fotoğrafı yok, ama üretilebilir. Mevcut fotoğrafların
bozulmuş sürümleri (döndürme, kırpma, parlaklık, yeniden sıkıştırma) "aynı ürün,
farklı çekim" durumunu taklit ediyor. Ölçülen iki dağılım:

  POZİTİF  aynı ürünün bozulmuş sürümü ile özgün hâli arasındaki benzerlik
  NEGATİF  AYNI KATEGORİDEKİ farklı ürünler arasındaki benzerlik
           (rastgele çift değil — zor negatif, çünkü asıl karışma riski burada)

İyi bir eşik, pozitiflerin altında kalmadan negatiflerin üstünde durmalı.

    .venv\\Scripts\\python.exe scripts\\test_kopya.py [urun_sayisi]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.models.embedder import GorselEmbedder  # noqa: E402

GECICI = config.VERI_DIZINI / "_kopya_testi"


def bozulmalar(gorsel: Image.Image) -> dict[str, Image.Image]:
    """Aynı ürünün farklı çekimini taklit eden makul bozulmalar."""
    g, y = gorsel.size
    return {
        "dondurme_8": gorsel.rotate(8, expand=True, fillcolor=(255, 255, 255)),
        "kirpma_%15": gorsel.crop((int(g * .08), int(y * .08), int(g * .92), int(y * .92))),
        "parlaklik_0.8": ImageEnhance.Brightness(gorsel).enhance(0.8),
        "parlaklik_1.25": ImageEnhance.Brightness(gorsel).enhance(1.25),
        "kucultme_50%": gorsel.resize((max(g // 2, 32), max(y // 2, 32))),
    }


def main() -> int:
    adet = int(sys.argv[1]) if len(sys.argv) > 1 else 12

    urunler = [
        json.loads(s)
        for s in (config.EVAL_DIZINI / "urunler.jsonl").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    # Kategori başına en az iki ürün olsun ki zor negatif çiftleri kurulabilsin.
    kategoriler: dict[str, list[dict]] = defaultdict(list)
    for u in urunler:
        kategoriler[u["kategori"]].append(u)
    secilen = [u for grup in kategoriler.values() for u in grup[:2]][:adet]

    print(f"[i] {len(secilen)} ürün, ürün başına {len(bozulmalar(Image.new('RGB', (8, 8))))} bozulma")
    GECICI.mkdir(parents=True, exist_ok=True)
    embedder = GorselEmbedder()

    # --- Özgün vektörler ---
    ozgun_yollar = [config.FOTO_DIZINI / u["dosya"] for u in secilen]
    ozgun = embedder.gorselleri_gom(ozgun_yollar, ilerleme=False)

    # --- Bozulmuş sürümler ---
    pozitif: list[tuple[str, str, float]] = []
    for u, vektor in zip(secilen, ozgun):
        gorsel = Image.open(config.FOTO_DIZINI / u["dosya"]).convert("RGB")
        yollar, adlar = [], []
        for ad, bozuk in bozulmalar(gorsel).items():
            hedef = GECICI / f"{Path(u['dosya']).stem}_{ad}.jpg"
            bozuk.save(hedef, "JPEG", quality=88)
            yollar.append(hedef)
            adlar.append(ad)
        bozuk_vektorler = embedder.gorselleri_gom(yollar, ilerleme=False)
        for ad, bv in zip(adlar, bozuk_vektorler):
            pozitif.append((u["dosya"], ad, float(np.dot(vektor, bv))))

    # --- Zor negatifler: aynı kategoride farklı ürünler ---
    negatif: list[tuple[str, str, float]] = []
    for i, ui in enumerate(secilen):
        for j, uj in enumerate(secilen):
            if i < j and ui["kategori"] == uj["kategori"]:
                negatif.append((ui["dosya"], uj["dosya"], float(np.dot(ozgun[i], ozgun[j]))))

    # --- Kolay negatifler: farklı kategoriler ---
    kolay: list[float] = [
        float(np.dot(ozgun[i], ozgun[j]))
        for i in range(len(secilen)) for j in range(i + 1, len(secilen))
        if secilen[i]["kategori"] != secilen[j]["kategori"]
    ]

    def ozet(ad: str, degerler: list[float]) -> None:
        if not degerler:
            print(f"  {ad:<26} (veri yok)")
            return
        d = np.array(degerler)
        print(f"  {ad:<26} n={len(d):<4} min={d.min():.3f}  "
              f"ort={d.mean():.3f}  max={d.max():.3f}  p5={np.percentile(d, 5):.3f}")

    print(f"\n{'=' * 74}")
    print("BENZERLİK DAĞILIMLARI")
    print("=" * 74)
    ozet("POZİTİF (aynı ürün)", [p[2] for p in pozitif])
    ozet("ZOR NEGATİF (ayn. kat.)", [n[2] for n in negatif])
    ozet("KOLAY NEGATİF (farklı kat.)", kolay)

    print(f"\n=== Bozulma türüne göre pozitifler ===")
    tur: dict[str, list[float]] = defaultdict(list)
    for _d, ad, s in pozitif:
        tur[ad].append(s)
    for ad, degerler in sorted(tur.items(), key=lambda x: np.mean(x[1])):
        d = np.array(degerler)
        print(f"  {ad:<18} min={d.min():.3f}  ort={d.mean():.3f}")

    # --- Eşik önerisi ---
    p = np.array([x[2] for x in pozitif])
    n = np.array([x[2] for x in negatif]) if negatif else np.array([0.0])
    print(f"\n{'=' * 74}")
    print("EŞİK ÖNERİSİ")
    print("=" * 74)
    print(f"  Pozitiflerin en düşüğü      : {p.min():.3f}")
    print(f"  Zor negatiflerin en yükseği : {n.max():.3f}")
    if p.min() > n.max():
        onerilen = round((p.min() + n.max()) / 2, 2)
        print(f"  Dağılımlar AYRIK. Önerilen AYNI_URUN_ESIGI = {onerilen}")
    else:
        # Örtüşme varsa yanlış birleştirmeyi önlemek esas: eşiği negatiflerin
        # üstüne koyup bazı kopyaları kaçırmak, farklı ürünleri birleştirmekten iyidir.
        onerilen = round(float(np.percentile(n, 99)) + 0.01, 2)
        kacan = float((p < onerilen).mean())
        print(f"  Dağılımlar ÖRTÜŞÜYOR.")
        print(f"  Yanlış birleştirmeyi önceleyen eşik = {onerilen}")
        print(f"  Bu eşikte kaçırılan kopya oranı: {kacan:.1%}")
    print(f"  BENZER_URUN_ESIGI için zor negatiflerin ortalaması: {n.mean():.3f}")

    # Geçici dosyaları temizle
    for f in GECICI.glob("*.jpg"):
        f.unlink()
    GECICI.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
