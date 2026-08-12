"""Retrieval ölçümü — indeks yapılandırmalarını yan yana karşılaştırır.

Tek çalıştırmada üç yapılandırma ölçülüyor:

  A      görsel indeks tek başına   (10-11 Ağustos taban çizgisi)
  B      metin indeksi tek başına   (VLM açıklamaları)
  A+B    RRF ile birleştirilmiş

Amaç sadece "iyileşti mi" değil, KAZANCIN NEREDEN GELDİĞİNİ görmek. Sorgu tipleri
ayrı raporlanıyor çünkü her indeksin güçlü olduğu yer farklı:

  kategori        kolay, görsel indeks çözmeli
  renk+kategori   ZOR, projenin asıl iddiası — B'nin kapatması beklenen açık
  marka           anahtar kelime işi, ikisi de zayıf (İndeks C bunun için gelecek)

    .venv\\Scripts\\python.exe eval\\olcum.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma  # noqa: E402
from src.search.arama import Arayici  # noqa: E402

K_DEGERLERI = (1, 5)
TIPLER = ("kategori", "renk+kategori", "marka")

YAPILANDIRMALAR = {
    "A (gorsel)": ("gorsel",),
    "B (metin)": ("metin",),
    "A+B (RRF)": ("gorsel", "metin"),
}


def sorgulari_yukle() -> list[dict]:
    yol = config.EVAL_DIZINI / "sorgular.jsonl"
    if not yol.exists():
        print(f"[!] Sorgu dosyası yok: {yol}")
        sys.exit(1)
    return [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def kimlik_haritasi() -> dict[str, int]:
    """ChromaDB kimliği -> veri seti ürün kimliği.

    İndeks kimlikleri fotoğrafın içerik özeti; ground truth ise veri setinin
    sayısal kimliğini kullanıyor. Eşleştirme metadata'daki dosya adı üzerinden.
    """
    kayit = chroma.gorsel_koleksiyon().get()
    return {
        kid: int(Path(m["dosya"]).stem)
        for kid, m in zip(kayit["ids"], kayit["metadatas"])
    }


def olc(arayici: Arayici, sorgular: list[dict], kullan: tuple[str, ...],
        harita: dict[str, int]) -> list[dict]:
    en_buyuk_k = max(K_DEGERLERI)
    sonuclar = []
    for s in sorgular:
        bulunan_kimlik = [
            r["kimlik"] for r in arayici.ara(s["sorgu"], k=en_buyuk_k, kullan=kullan)
        ]
        bulunan = [harita[k] for k in bulunan_kimlik if k in harita]
        dogru = set(s["dogru"])
        ilk_sira = next((i for i, b in enumerate(bulunan, 1) if b in dogru), None)
        satir = {
            "sorgu": s["sorgu"],
            "tip": s["tip"],
            "rr": 1.0 / ilk_sira if ilk_sira else 0.0,
            "ilk_sira": ilk_sira,
        }
        for k in K_DEGERLERI:
            satir[f"recall@{k}"] = len(dogru & set(bulunan[:k])) / len(dogru)
        sonuclar.append(satir)
    return sonuclar


def ort(kayitlar: list[dict], alan: str) -> float:
    return sum(k[alan] for k in kayitlar) / len(kayitlar) if kayitlar else 0.0


def main() -> int:
    sorgular = sorgulari_yukle()
    gorsel_sayi = chroma.gorsel_koleksiyon().count()
    metin_sayi = chroma.metin_koleksiyon().count()

    if gorsel_sayi == 0:
        print("[!] Görsel indeks boş. Önce: scripts\\faz0.py ekle data\\photos")
        return 1
    print(f"[i] İndeks A: {gorsel_sayi} kayıt · İndeks B: {metin_sayi} kayıt")
    print(f"[i] {len(sorgular)} sorgu\n")

    harita = kimlik_haritasi()
    arayici = Arayici(metin=metin_sayi > 0)

    tum: dict[str, list[dict]] = {}
    for ad, kullan in YAPILANDIRMALAR.items():
        if "metin" in kullan and metin_sayi == 0:
            continue
        print(f"[i] Ölçülüyor: {ad}")
        tum[ad] = olc(arayici, sorgular, kullan, harita)

    # --- Karşılaştırma tablosu ---
    print(f"\n{'=' * 78}")
    print("KARŞILAŞTIRMA — sorgu tipine göre")
    print("=" * 78)
    for tip in TIPLER:
        alt_sayi = len([s for s in sorgular if s["tip"] == tip])
        print(f"\n{tip}  ({alt_sayi} sorgu)")
        print(f"  {'yapılandırma':<16}{'R@1':>10}{'R@5':>10}{'MRR':>10}")
        print("  " + "-" * 46)
        for ad, sonuc in tum.items():
            alt = [s for s in sonuc if s["tip"] == tip]
            print(f"  {ad:<16}{ort(alt, 'recall@1'):>10.3f}"
                  f"{ort(alt, 'recall@5'):>10.3f}{ort(alt, 'rr'):>10.3f}")

    print(f"\n{'=' * 78}")
    print("GENEL")
    print("=" * 78)
    print(f"  {'yapılandırma':<16}{'R@1':>10}{'R@5':>10}{'MRR':>10}{'ıskalanan':>12}")
    print("  " + "-" * 58)
    for ad, sonuc in tum.items():
        iskalanan = len([s for s in sonuc if s["ilk_sira"] is None])
        print(f"  {ad:<16}{ort(sonuc, 'recall@1'):>10.3f}"
              f"{ort(sonuc, 'recall@5'):>10.3f}{ort(sonuc, 'rr'):>10.3f}"
              f"{iskalanan:>8}/{len(sonuc)}")

    # --- Kazanç / kayıp ---
    if "A (gorsel)" in tum and "A+B (RRF)" in tum:
        print(f"\n{'=' * 78}")
        print("A -> A+B DEĞİŞİM")
        print("=" * 78)
        taban, hibrit = tum["A (gorsel)"], tum["A+B (RRF)"]
        for tip in TIPLER:
            t = [s for s in taban if s["tip"] == tip]
            h = [s for s in hibrit if s["tip"] == tip]
            d1 = ort(h, "recall@1") - ort(t, "recall@1")
            d5 = ort(h, "recall@5") - ort(t, "recall@5")
            print(f"  {tip:<16} R@1 {d1:+.3f}   R@5 {d5:+.3f}")
        d1 = ort(hibrit, "recall@1") - ort(taban, "recall@1")
        d5 = ort(hibrit, "recall@5") - ort(taban, "recall@5")
        print(f"  {'TÜMÜ':<16} R@1 {d1:+.3f}   R@5 {d5:+.3f}")

        # Hibritte düzelen ve bozulan sorgular
        duzelen = [h["sorgu"] for t, h in zip(taban, hibrit)
                   if t["ilk_sira"] is None and h["ilk_sira"] is not None]
        bozulan = [h["sorgu"] for t, h in zip(taban, hibrit)
                   if t["ilk_sira"] is not None and h["ilk_sira"] is None]
        print(f"\n  Hibritte DÜZELEN sorgu: {len(duzelen)}")
        for s in duzelen[:10]:
            print(f"    + {s}")
        print(f"  Hibritte BOZULAN sorgu: {len(bozulan)}")
        for s in bozulan[:10]:
            print(f"    - {s}")

    cikti = config.EVAL_DIZINI / "sonuc_karsilastirma.json"
    cikti.write_text(
        json.dumps(
            {
                "urun_sayisi": gorsel_sayi,
                "sorgu_sayisi": len(sorgular),
                "yapilandirmalar": {
                    ad: {
                        "genel": {f"recall@{k}": ort(s, f"recall@{k}") for k in K_DEGERLERI}
                        | {"mrr": ort(s, "rr")},
                        "tipe_gore": {
                            tip: {f"recall@{k}": ort([x for x in s if x["tip"] == tip], f"recall@{k}")
                                  for k in K_DEGERLERI}
                            | {"mrr": ort([x for x in s if x["tip"] == tip], "rr")}
                            for tip in TIPLER
                        },
                        "sorgular": s,
                    }
                    for ad, s in tum.items()
                },
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[+] Ayrıntılı sonuç: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
