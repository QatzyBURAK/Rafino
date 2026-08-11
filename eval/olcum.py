"""Retrieval ölçümü — Recall@1, Recall@5, MRR.

Projenin iddiasını sayıya çevirir. Her model veya mimari değişikliğinden sonra
çalıştırılır ve sonuç deftere yazılır; böylece "iyileşti" iddiası ölçüye dayanır.

Sorgu tipleri ayrı ayrı raporlanıyor, çünkü hangi indeksin ne kazandırdığı
ancak böyle görülüyor:

  kategori        -> kolay. Görsel indeks tek başına çözmeli.
  renk+kategori   -> ZOR. Projenin asıl iddiası. Görsel indeksin zayıf olduğu yer.
  marka           -> anahtar kelime işi. Vektör aramasının kötü olduğu yer.

    .venv\\Scripts\\python.exe eval\\olcum.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma  # noqa: E402
from src.models.embedder import GorselEmbedder  # noqa: E402

K_DEGERLERI = (1, 5)


def sorgulari_yukle() -> list[dict]:
    yol = config.EVAL_DIZINI / "sorgular.jsonl"
    if not yol.exists():
        print(f"[!] Sorgu dosyası yok: {yol}")
        print("    Önce: .venv\\Scripts\\python.exe scripts\\veriseti_kur.py")
        sys.exit(1)
    return [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def main() -> int:
    sorgular = sorgulari_yukle()
    koleksiyon = chroma.gorsel_koleksiyon()
    kayitli = koleksiyon.count()
    if kayitli == 0:
        print("[!] Görsel indeks boş. Önce: scripts\\faz0.py ekle data\\photos")
        return 1

    print(f"[i] İndekste {kayitli} ürün, {len(sorgular)} sorgu ölçülecek")
    print(f"[i] Aranan indeks: A (görsel) — tek indeksli taban çizgisi\n")

    embedder = GorselEmbedder()
    en_buyuk_k = max(K_DEGERLERI)

    # Sorguları toplu göm: tek tek gömmek yavaş ve gereksiz.
    metinler = [s["sorgu"] for s in sorgular]
    vektorler = embedder.sorguyu_gom(metinler, ilerleme=True)

    sonuclar: list[dict] = []
    for sorgu, vektor in zip(sorgular, vektorler):
        cevap = koleksiyon.query(
            query_embeddings=[vektor.tolist()], n_results=en_buyuk_k
        )
        # ChromaDB kimliği içerik özeti; ground truth veri seti kimliği.
        # Eşleştirme metadata'daki dosya adı üzerinden yapılıyor.
        bulunan = [
            int(Path(m["dosya"]).stem) for m in cevap["metadatas"][0]
        ]
        dogru = set(sorgu["dogru"])

        # İlk doğru cevabın sırası (MRR için)
        ilk_sira = next(
            (i for i, b in enumerate(bulunan, 1) if b in dogru), None
        )
        satir = {
            "sorgu": sorgu["sorgu"],
            "tip": sorgu["tip"],
            "zorluk": sorgu["zorluk"],
            "dogru_sayisi": len(dogru),
            "rr": 1.0 / ilk_sira if ilk_sira else 0.0,
            "ilk_sira": ilk_sira,
        }
        for k in K_DEGERLERI:
            isabet = len(dogru & set(bulunan[:k]))
            # Recall: doğru cevapların kaçı ilk k'da. Tek doğrulu sorguda
            # bu isabet oranına eşit oluyor.
            satir[f"recall@{k}"] = isabet / len(dogru)
        sonuclar.append(satir)

    def ortalama(kayitlar: list[dict], alan: str) -> float:
        return sum(k[alan] for k in kayitlar) / len(kayitlar) if kayitlar else 0.0

    print(f"\n{'=' * 66}")
    print("SONUÇ — İndeks A tek başına (taban çizgisi)")
    print("=" * 66)
    print(f"{'sorgu tipi':<18}{'adet':>6}{'R@1':>9}{'R@5':>9}{'MRR':>9}")
    print("-" * 66)

    for tip in ["kategori", "renk+kategori", "marka"]:
        alt = [s for s in sonuclar if s["tip"] == tip]
        if not alt:
            continue
        print(f"{tip:<18}{len(alt):>6}"
              f"{ortalama(alt, 'recall@1'):>9.3f}"
              f"{ortalama(alt, 'recall@5'):>9.3f}"
              f"{ortalama(alt, 'rr'):>9.3f}")

    print("-" * 66)
    print(f"{'TÜMÜ':<18}{len(sonuclar):>6}"
          f"{ortalama(sonuclar, 'recall@1'):>9.3f}"
          f"{ortalama(sonuclar, 'recall@5'):>9.3f}"
          f"{ortalama(sonuclar, 'rr'):>9.3f}")

    # En kötü sorgular: neyin bozuk olduğunu gösteren asıl bilgi burada.
    kotu = [s for s in sonuclar if s["ilk_sira"] is None]
    print(f"\n[i] İlk {en_buyuk_k} sonuçta hiç doğru cevap çıkmayan: "
          f"{len(kotu)}/{len(sonuclar)}")
    if kotu:
        print("\n=== Tamamen ıskalanan sorgular (ilk 15) ===")
        for s in kotu[:15]:
            print(f"  [{s['tip']:<14}] \"{s['sorgu']}\"")

    cikti = config.EVAL_DIZINI / "sonuc_indeks_a.json"
    cikti.write_text(
        json.dumps(
            {
                "indeks": "A (görsel) tek başına",
                "urun_sayisi": kayitli,
                "sorgu_sayisi": len(sonuclar),
                "ozet": {
                    f"recall@{k}": ortalama(sonuclar, f"recall@{k}")
                    for k in K_DEGERLERI
                }
                | {"mrr": ortalama(sonuclar, "rr")},
                "sorgular": sonuclar,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[+] Ayrıntılı sonuç: {cikti}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
