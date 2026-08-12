"""Renk çıkarımının doğruluğunu ground truth'a karşı ölçer.

Değerlendirme setindeki 60 ürünün gerçek renk etiketi zaten elimizde olduğu için
piksel tabanlı çıkarım doğrudan ölçülebiliyor. İki oran raporlanıyor:

  birebir  — tahmin, etiketle tam aynı
  makul    — tahmin farklı ama yakın renk grubunda (gümüş->gri, altın->sarı gibi)

"makul" gevşeklik değil bir tasarım kararı: palet bilerek kabalaştırıldı, çünkü
gümüş ile griyi pikselden ayırmak güvenilir değil ve depo aramasında bu ayrımın
karşılığı yok.

    .venv\\Scripts\\python.exe scripts\\test_renk.py
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.ingest.renk import makul_mu, renk_bul  # noqa: E402


def main() -> int:
    yol = config.EVAL_DIZINI / "urunler.jsonl"
    if not yol.exists():
        print(f"[!] Ürün dosyası yok: {yol}")
        return 1
    urunler = [json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]
    print(f"[i] {len(urunler)} üründe renk çıkarımı ölçülüyor\n")

    birebir = makul = 0
    hatalar: list[tuple] = []
    karisiklik: Counter = Counter()
    t0 = time.perf_counter()

    for u in urunler:
        foto = config.FOTO_DIZINI / u["dosya"]
        if not foto.exists():
            continue
        tahmin, pay = renk_bul(foto)
        gercek = u["renk"]
        if tahmin == gercek:
            birebir += 1
            makul += 1
        elif makul_mu(tahmin, gercek):
            makul += 1
            karisiklik[f"{gercek} -> {tahmin}"] += 1
        else:
            hatalar.append((u["dosya"], u["kategori"], gercek, tahmin, pay, u["urun_adi"]))
            karisiklik[f"{gercek} -> {tahmin}"] += 1

    sure = time.perf_counter() - t0
    n = len(urunler)

    print("=" * 62)
    print("RENK ÇIKARIMI — piksel tabanlı, deterministik")
    print("=" * 62)
    print(f"  birebir doğru : {birebir}/{n}  ({birebir / n:.1%})")
    print(f"  makul doğru   : {makul}/{n}  ({makul / n:.1%})")
    print(f"  gerçek hata   : {len(hatalar)}/{n}  ({len(hatalar) / n:.1%})")
    print(f"  hız           : {sure / n * 1000:.0f} ms/fotoğraf  (VLM ~4500 ms)")

    if karisiklik:
        print(f"\n=== En sık karışıklıklar ===")
        for cift, adet in karisiklik.most_common(10):
            print(f"  {cift:<34} {adet}")

    if hatalar:
        print(f"\n=== Gerçek hatalar ===")
        print(f"{'dosya':<12}{'kategori':<18}{'etiket':<14}{'tahmin':<14}{'pay':<7}ürün")
        print("-" * 100)
        for dosya, kat, gercek, tahmin, pay, ad in hatalar[:20]:
            print(f"{dosya:<12}{kat:<18}{gercek:<14}{tahmin:<14}{pay:<7.2f}{ad[:38]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
