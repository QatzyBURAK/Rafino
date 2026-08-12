"""Marka çıkarımının kesinliğini ve kapsamını ayrı ayrı ölçer.

İki farklı soru var ve karıştırılmamalı:

  kapsam (recall)   — VLM kaç üründe marka okuyabildi?
  kesinlik (precision) — okuduğunu iddia ettiğinde ne kadar doğru?

Bu ayrım tasarım kararını belirliyor. Kesinlik yüksek ama kapsam düşükse model
"uydurmuyor, sadece göremiyor" demektir; o zaman doğru çözüm modeli zorlamak
değil, okunamayan markalar için elle giriş alanı açmaktır.

    .venv\\Scripts\\python.exe scripts\\test_marka.py
"""

import json
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402

BILINMEYEN = {"bilinmiyor", "bilinmeyen", "yok", "unknown", "none", "", "-"}


def sadelestir(metin: str) -> str:
    """Marka karşılaştırması için: büyük/küçük, aksan ve noktalama farkını sil."""
    metin = unicodedata.normalize("NFKD", str(metin).lower().strip())
    return "".join(c for c in metin if c.isalnum())


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

    okudu_dogru: list[tuple] = []
    okudu_kismi: list[tuple] = []
    okudu_yanlis: list[tuple] = []
    okuyamadi: list[tuple] = []

    for dosya in ortak:
        gercek = urunler[dosya]["marka"]
        tahmin = str(vlm[dosya].get("marka", "")).strip()
        s_gercek, s_tahmin = sadelestir(gercek), sadelestir(tahmin)

        if s_tahmin in {sadelestir(x) for x in BILINMEYEN}:
            okuyamadi.append((dosya, gercek, urunler[dosya]["urun_adi"]))
        elif s_tahmin == s_gercek:
            okudu_dogru.append((dosya, gercek, tahmin))
        elif s_tahmin and (s_tahmin in s_gercek or s_gercek in s_tahmin):
            # Kısmi eşleşme: "Formula 1" vs "Formula 1 Go", "nike" vs
            # "Nike Fragrances". Marka doğru tanınmış, yalnızca ürün serisi
            # eklenmiş/eksik. Stok aramasında bunlar aynı markadır.
            okudu_kismi.append((dosya, gercek, tahmin))
        else:
            okudu_yanlis.append((dosya, gercek, tahmin, urunler[dosya]["urun_adi"]))

    n = len(ortak)
    okudu = len(okudu_dogru) + len(okudu_kismi) + len(okudu_yanlis)
    kabul = len(okudu_dogru) + len(okudu_kismi)

    print("=" * 64)
    print("MARKA ÇIKARIMI — kapsam ve kesinlik")
    print("=" * 64)
    print(f"  Toplam ürün            : {n}")
    print(f"  Marka okuduğunu iddia  : {okudu}/{n}  ({okudu / n:.1%})   <- kapsam")
    print(f"  'bilinmiyor' dedi      : {len(okuyamadi)}/{n}  ({len(okuyamadi) / n:.1%})")
    print()
    if okudu:
        print(f"  Okuduklarında birebir  : {len(okudu_dogru)}/{okudu}  "
              f"({len(okudu_dogru) / okudu:.1%})")
        print(f"  Okuduklarında kısmi    : {len(okudu_kismi)}/{okudu}  "
              f"({len(okudu_kismi) / okudu:.1%})   <- ayni marka, farkli yazim")
        print(f"  KESİNLİK (birebir+kısmi): {kabul}/{okudu}  ({kabul / okudu:.1%})")
        print(f"  Gerçekten yanlış       : {len(okudu_yanlis)}/{okudu}  "
              f"({len(okudu_yanlis) / okudu:.1%})   <- uydurma")
    print(f"\n  Tüm ürünlerde doğru    : {kabul}/{n}  ({kabul / n:.1%})")

    if okudu_kismi:
        print(f"\n=== Kısmi eşleşenler (doğru sayıldı) ===")
        for _d, gercek, tahmin in okudu_kismi:
            print(f"  {gercek:<22} <- VLM: {tahmin}")

    if okudu_yanlis:
        print(f"\n=== Yanlış okunanlar (uydurma) ===")
        print(f"{'gerçek':<16}{'VLM':<20}ürün")
        print("-" * 84)
        for _d, gercek, tahmin, ad in okudu_yanlis[:20]:
            print(f"{gercek:<16}{tahmin:<20}{ad[:44]}")

    if okuyamadi:
        print(f"\n=== 'bilinmiyor' denen ürünlerin gerçek markaları ===")
        for _d, gercek, ad in okuyamadi[:20]:
            print(f"  {gercek:<16}{ad[:52]}")

    print("\n" + "=" * 64)
    if okudu and kabul / okudu >= 0.85:
        print("SONUÇ: Kesinlik yüksek — model uydurmuyor, sadece göremiyor.")
        print("       Doğru çözüm modeli zorlamak değil, okunamayan markalar için")
        print("       elle giriş alanı açmak ve kaynağı kayıtta tutmak.")
    else:
        print("SONUÇ: Kesinlik düşük — model okuyamadığında uyduruyor.")
        print("       Prompt sıkılaştırılmalı, 'bilinmiyor' demesi teşvik edilmeli.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
