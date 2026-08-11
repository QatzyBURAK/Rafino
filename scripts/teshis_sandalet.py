"""Belirli bir kategorinin neden ıskalandığını teşhis eder.

Üç ihtimali ayırt eder:
  1) Model Türkçe kelimeyi tanımıyor      -> İngilizce sorgu düzeltir
  2) Ürünler görsel olarak benzeşiyor      -> yerine gelen sonuçlar akraba kategoriden
  3) Sorgu ile görsel arası bağ zayıf      -> hiçbir varyant işe yaramaz

    .venv\\Scripts\\python.exe scripts\\teshis_sandalet.py [kategori]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma  # noqa: E402
from src.models.embedder import GorselEmbedder  # noqa: E402

# Aynı kavramın farklı ifadeleri. Hangisinin tuttuğu teşhisi veriyor.
VARYANTLAR = {
    "sandalet": ["sandalet", "sandaletler", "sandals", "sandal",
                 "acik ayakkabi", "yazlik ayakkabi", "kadin sandalet"],
    "gömlek": ["gömlek", "gomlek", "shirt", "erkek gömleği", "uzun kollu gömlek"],
}


def main() -> int:
    kategori = sys.argv[1] if len(sys.argv) > 1 else "sandalet"

    urunler = [
        json.loads(s)
        for s in (config.EVAL_DIZINI / "urunler.jsonl").read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    kimlik_kategori = {u["id"]: u["kategori"] for u in urunler}
    hedef_idler = {u["id"] for u in urunler if u["kategori"] == kategori}
    if not hedef_idler:
        print(f"[!] '{kategori}' kategorisi sette yok")
        return 1

    print(f"[i] '{kategori}' kategorisinde {len(hedef_idler)} ürün var")
    print(f"[i] Kimlikler: {sorted(hedef_idler)}\n")

    koleksiyon = chroma.gorsel_koleksiyon()
    embedder = GorselEmbedder()

    varyantlar = VARYANTLAR.get(kategori, [kategori])
    vektorler = embedder.sorguyu_gom(varyantlar)

    print(f"{'sorgu':<24}{'hedef ilk sıra':<16}ilk 5 sonucun kategorileri")
    print("-" * 92)

    for sorgu, vektor in zip(varyantlar, vektorler):
        cevap = koleksiyon.query(query_embeddings=[vektor.tolist()], n_results=10)
        bulunan = [int(Path(m["dosya"]).stem) for m in cevap["metadatas"][0]]
        ilk_sira = next((i for i, b in enumerate(bulunan, 1) if b in hedef_idler), None)
        ilk5 = ", ".join(kimlik_kategori.get(b, "?") for b in bulunan[:5])
        sira_str = str(ilk_sira) if ilk_sira else "yok (10'da)"
        print(f"{sorgu:<24}{sira_str:<16}{ilk5}")

    # Hedef ürünlerin her biri kendi kategorisiyle ne kadar örtüşüyor?
    print(f"\n=== '{kategori}' ürünleri, kendi kategori sorgusundaki sıraları ===")
    vektor = embedder.sorguyu_gom([kategori])[0]
    cevap = koleksiyon.query(query_embeddings=[vektor.tolist()], n_results=60)
    bulunan = [int(Path(m["dosya"]).stem) for m in cevap["metadatas"][0]]
    mesafeler = cevap["distances"][0]
    for hid in sorted(hedef_idler):
        if hid in bulunan:
            sira = bulunan.index(hid) + 1
            benzerlik = 1 - mesafeler[sira - 1]
            urun = next(u for u in urunler if u["id"] == hid)
            print(f"  {hid}  sıra {sira:>2}/60  benzerlik {benzerlik:.3f}  "
                  f"({urun['renk']}, {urun['urun_adi'][:45]})")

    print(f"\n=== '{kategori}' sorgusunda ilk 5'i kapan ürünler ===")
    for i, (b, m) in enumerate(zip(bulunan[:5], mesafeler[:5]), 1):
        urun = next((u for u in urunler if u["id"] == b), None)
        if urun:
            print(f"  {i}. {1 - m:.3f}  [{urun['kategori']}] {urun['urun_adi'][:55]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
