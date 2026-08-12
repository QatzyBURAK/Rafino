"""K4 kararını sınar: rengi piksellerden mi VLM'den mi almalıyız?

K4 (10 Ağustos) bir GEREKÇEYE dayanıyordu, ölçüme değil: VLM aynı çağrıda hem
alan hem serbest metin istendiğinde kendi içinde çelişiyordu, o yüzden renk
pikselden hesaplansın denmişti.

Elimizde 60 ürünün gerçek renk etiketi olduğu için bu varsayım artık
sınanabilir. İki yöntem aynı veride, aynı ölçütle karşılaştırılıyor.

    .venv\\Scripts\\python.exe scripts\\karsilastir_renk.py
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.ingest.renk import birebir_mi, makul_mu, renk_bul  # noqa: E402


def main() -> int:
    urun_yolu = config.EVAL_DIZINI / "urunler.jsonl"
    vlm_yolu = config.VERI_DIZINI / "oznitelikler.jsonl"
    for y in (urun_yolu, vlm_yolu):
        if not y.exists():
            print(f"[!] Dosya yok: {y}")
            return 1

    urunler = {
        json.loads(s)["dosya"]: json.loads(s)
        for s in urun_yolu.read_text(encoding="utf-8").splitlines() if s.strip()
    }
    vlm = {
        json.loads(s)["dosya"]: json.loads(s)
        for s in vlm_yolu.read_text(encoding="utf-8").splitlines() if s.strip()
    }

    ortak = sorted(set(urunler) & set(vlm))
    print(f"[i] {len(ortak)} üründe iki yöntem karşılaştırılıyor\n")

    sonuc = {"piksel": {"birebir": 0, "makul": 0}, "vlm": {"birebir": 0, "makul": 0}}
    ayrisma: list[tuple] = []
    vlm_karisiklik: Counter = Counter()

    for dosya in ortak:
        gercek = urunler[dosya]["renk"]
        piksel_renk, _ = renk_bul(config.FOTO_DIZINI / dosya)
        vlm_renk = str(vlm[dosya].get("renk", "")).strip().lower()

        for ad, tahmin in (("piksel", piksel_renk), ("vlm", vlm_renk)):
            if birebir_mi(tahmin, gercek):
                sonuc[ad]["birebir"] += 1
                sonuc[ad]["makul"] += 1
            elif makul_mu(tahmin, gercek):
                sonuc[ad]["makul"] += 1

        if vlm_renk != piksel_renk:
            p_ok = makul_mu(piksel_renk, gercek)
            v_ok = makul_mu(vlm_renk, gercek)
            if p_ok != v_ok:
                ayrisma.append((dosya, gercek, piksel_renk, vlm_renk,
                                "VLM" if v_ok else "piksel",
                                urunler[dosya]["urun_adi"]))
        if not makul_mu(vlm_renk, gercek):
            vlm_karisiklik[f"{gercek} -> {vlm_renk}"] += 1

    n = len(ortak)
    print("=" * 60)
    print("K4 SINAMASI — renk kaynağı karşılaştırması")
    print("=" * 60)
    print(f"{'yöntem':<12}{'birebir':>16}{'makul':>16}")
    print("-" * 60)
    for ad in ("piksel", "vlm"):
        b, m = sonuc[ad]["birebir"], sonuc[ad]["makul"]
        print(f"{ad:<12}{b:>7}/{n} {b / n:>6.1%}{m:>7}/{n} {m / n:>6.1%}")

    fark = sonuc["vlm"]["makul"] - sonuc["piksel"]["makul"]
    print("-" * 60)
    print(f"Fark (makul): VLM {'+' if fark >= 0 else ''}{fark} ürün "
          f"({fark / n:+.1%})")

    if vlm_karisiklik:
        print(f"\n=== VLM'in hataları ===")
        for cift, adet in vlm_karisiklik.most_common(10):
            print(f"  {cift:<34} {adet}")

    if ayrisma:
        print(f"\n=== İki yöntemin ayrıştığı ürünler ({len(ayrisma)}) ===")
        print(f"{'etiket':<13}{'piksel':<13}{'vlm':<13}{'kazanan':<9}ürün")
        print("-" * 96)
        for _dosya, gercek, p, v, kazanan, ad in ayrisma[:20]:
            print(f"{gercek:<13}{p:<13}{v:<13}{kazanan:<9}{ad[:40]}")

    print("\n" + "=" * 60)
    if fark > 3:
        print("SONUÇ: VLM belirgin şekilde daha iyi. K4 tersine çevrilmeli.")
    elif fark < -3:
        print("SONUÇ: Piksel daha iyi. K4 doğrulandı, olduğu gibi kalsın.")
    else:
        print("SONUÇ: Fark küçük. Piksel yöntemi tercih edilir — deterministik,")
        print("       180 kat hızlı (25 ms / 4500 ms) ve VLM çağrısını kısaltır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
