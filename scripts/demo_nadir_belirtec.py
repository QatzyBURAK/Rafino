"""Nadir belirteç problemini ölçerek gösterir.

İki şey ortaya koyuyor:
  1. Kelimeler modele nasıl parçalanıyor (tokenization)
  2. Nadir kelimelerin vektörleri neden güvenilmez

    .venv\Scripts\python.exe scripts\demo_nadir_belirtec.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src import config

KELIMELER = [
    "mavi", "çanta", "kemer", "ayakkabı",
    "Puma", "Nike", "Titan", "Fossil",
    "Samsonite", "Samsung",
    "Lino Perros", "Louis Philippe", "Peter England", "John Miller",
    "SM-A546B", "SM-A546C",
]

CIFTLER = [
    ("mavi", "lacivert", "yakın OLMALI - gerçekten ilişkili"),
    ("çanta", "kemer", "orta - ikisi de aksesuar"),
    ("Samsonite", "Samsung", "UZAK olmalı - alakasız markalar"),
    ("Lino Perros", "Louis Philippe", "UZAK olmalı - farklı markalar"),
    ("Peter England", "John Miller", "UZAK olmalı - farklı markalar"),
    ("SM-A546B", "SM-A546C", "UZAK olmalı - farklı ürün kodları"),
    ("Puma", "Nike", "UZAK olmalı - rakip markalar"),
]


def main() -> int:
    from transformers import AutoTokenizer
    from src.models.embedder import MetinEmbedder

    print("=" * 70)
    print("1. KELİMELER MODELE NASIL PARÇALANIYOR")
    print("=" * 70)
    tok = AutoTokenizer.from_pretrained(config.METIN_MODEL)
    print(f"{'kelime':<18}{'parça':<4}parçalar")
    print("-" * 70)
    for k in KELIMELER:
        parcalar = tok.tokenize(k)
        print(f"{k:<18}{len(parcalar):<4}{' | '.join(parcalar)}")

    print()
    print("=" * 70)
    print("2. VEKTÖR UZAYINDA YAKINLIK (cosine)")
    print("=" * 70)
    emb = MetinEmbedder()
    hepsi = sorted({x for c in CIFTLER for x in c[:2]})
    vektorler = dict(zip(hepsi, emb.belgeleri_gom(hepsi)))

    print(f"{'çift':<34}{'cos':>7}   beklenti")
    print("-" * 70)
    for a, b, beklenti in CIFTLER:
        s = float(np.dot(vektorler[a], vektorler[b]))
        isaret = "  <-- SORUN" if ("UZAK" in beklenti and s > 0.85) else ""
        print(f"{a + ' / ' + b:<34}{s:>7.3f}   {beklenti}{isaret}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
