"""Veri seti keşfi — hangi kategori ve renkler var, hangi kombinasyonlar bol?

Değerlendirme setinin kalitesi tek bir şeye bağlı: AYNI kategoride FARKLI renkli
ürün çiftleri bulabilmek. "mavi el çantası" sorgusunun ayırt edici olması için
stokta siyah el çantası da olmalı. Bu betik o kombinasyonları sayar.

    .venv\\Scripts\\python.exe scripts\\kesif_veriseti.py
"""

import os
from collections import Counter

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

VERI_SETI = "PestoRosso/lamoda-fashion-product-images"
PARCA = "data/train_part_0000.parquet"


def main() -> int:
    print(f"[i] İndiriliyor: {VERI_SETI} / {PARCA}")
    yol = hf_hub_download(VERI_SETI, PARCA, repo_type="dataset")
    print(f"[+] {yol}")

    # Görselleri okumadan sadece etiket sütunlarını al — çok daha hızlı.
    tablo = pq.read_table(
        yol,
        columns=["id", "article_type", "base_color", "product_display_name",
                 "master_category", "width", "height"],
    )
    print(f"[i] Satır sayısı: {tablo.num_rows}")

    tipler = tablo.column("article_type").to_pylist()
    renkler = tablo.column("base_color").to_pylist()
    genisler = tablo.column("width").to_pylist()
    yuksekler = tablo.column("height").to_pylist()

    print(f"\n[i] Çözünürlük: en küçük {min(genisler)}x{min(yuksekler)}, "
          f"en büyük {max(genisler)}x{max(yuksekler)}")

    print(f"\n=== En sık 25 kategori ===")
    for tip, adet in Counter(tipler).most_common(25):
        print(f"  {tip:<28} {adet}")

    print(f"\n=== En sık 25 renk ===")
    for renk, adet in Counter(renkler).most_common(25):
        print(f"  {renk:<28} {adet}")

    # Asıl aradığımız: kaç farklı renkte bulunabilen kategoriler.
    kategori_renkleri: dict[str, Counter] = {}
    for tip, renk in zip(tipler, renkler):
        kategori_renkleri.setdefault(tip, Counter())[renk] += 1

    print(f"\n=== Renk çeşitliliği en yüksek kategoriler ===")
    print(f"{'kategori':<28}{'renk sayisi':<13}{'toplam':<9}en sık renkler")
    print("-" * 95)
    sirali = sorted(
        kategori_renkleri.items(),
        key=lambda x: (len(x[1]), sum(x[1].values())),
        reverse=True,
    )
    for tip, sayac in sirali[:20]:
        ilk = ", ".join(f"{r}({a})" for r, a in sayac.most_common(5))
        print(f"{tip:<28}{len(sayac):<13}{sum(sayac.values()):<9}{ilk}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
