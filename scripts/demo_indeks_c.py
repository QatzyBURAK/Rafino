"""İndeks C'nin (SQLite FTS5) içini ve çalışma biçimini gösterir."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import sqlite

def main() -> int:
    b = sqlite.baglan()

    print("=" * 72)
    print("1. TABLODA NE VAR (İndeks C'nin gördüğü alanlar)")
    print("=" * 72)
    print(f"{'marka':<18}{'kategori':<16}{'kaynak':<12}aciklama")
    print("-" * 72)
    for s in b.execute(
        "SELECT marka, kategori, marka_kaynagi, aciklama FROM urun LIMIT 6"
    ):
        print(f"{(s['marka'] or '-'):<18}{(s['kategori'] or '-'):<16}"
              f"{(s['marka_kaynagi'] or '-'):<12}{(s['aciklama'] or '')[:34]}")

    print()
    print("=" * 72)
    print("2. TERS İNDEKS (inverted index) — FTS5 içeride ne tutuyor")
    print("=" * 72)
    print("Kelime -> hangi kayıtlarda geçiyor. Örnek birkaç kelime:\n")
    for kelime in ("perros", "puma", "kemer", "deri"):
        satirlar = b.execute(
            "SELECT COUNT(*) FROM urun_fts WHERE urun_fts MATCH ?", (f'"{kelime}"',)
        ).fetchone()[0]
        print(f"  {kelime:<12} -> {satirlar} kayıtta geçiyor")

    print()
    print("=" * 72)
    print("3. VEKTÖRÜN ÇÖKTÜĞÜ SORGULAR — FTS5 NE YAPIYOR")
    print("=" * 72)
    testler = [
        ("Lino Perros", "vektörde Louis Philippe'e 0.899 yakın"),
        ("Peter England", "vektörde John Miller'a 0.887 yakın"),
        ("Puma", "vektörde Nike'a 0.900 yakın"),
    ]
    for sorgu, not_ in testler:
        kimlikler = sqlite.ara(b, sorgu, limit=3)
        if kimlikler:
            yer = ",".join("?" * len(kimlikler))
            satirlar = b.execute(
                f"SELECT marka, kategori FROM urun WHERE kimlik IN ({yer})", kimlikler
            ).fetchall()
            sonuc = " | ".join(f"{s['marka']} ({s['kategori']})" for s in satirlar)
        else:
            sonuc = "(sonuç yok)"
        print(f"\n  Sorgu: \"{sorgu}\"   [{not_}]")
        print(f"  FTS5  : {sonuc}")

    print()
    print("=" * 72)
    print("4. FTS5'İN YAPAMADIĞI — çekim eki")
    print("=" * 72)
    for sorgu in ("kemer", "kemerler", "kemeri", "kmer"):
        n = len(sqlite.ara(b, sorgu, limit=20))
        print(f"  \"{sorgu}\"{' ' * (12 - len(sorgu))}-> {n} sonuç")

    b.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
