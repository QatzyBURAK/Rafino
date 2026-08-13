"""İndeks C'yi kurar: SQLite + FTS5 anahtar kelime araması.

İki kip var ve aradaki fark K11 kararının (marka elle tamamlanabilsin)
değerini doğrudan ölçüyor:

    vlm   Yalnızca VLM'in okuyabildiği markalar indekslenir.
          Gerçekçi durum: ürünlerin %43'ünde marka var, %57'sinde yok.

    elle  Eksik markalar ground truth'tan doldurulur.
          Operatörün "bilinmiyor" kayıtları elle tamamladığı durumu taklit eder.
          Bu bir kopya çekme değil, K11 uygulandığında sistemin ULAŞACAĞI
          tavanı ölçmek; iki kipin farkı özelliğin kazancıdır.

FTS5'e ürün adı (product_display_name) KONULMUYOR — o ground truth ve
indekslenirse ölçüm anlamsızlaşır. Yalnızca sistemin gerçekten sahip olduğu
alanlar giriyor: marka, kategori, ürün kodu, üretilen açıklama.

    .venv\\Scripts\\python.exe scripts\\indeks_c_kur.py [vlm|elle]
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma, sqlite  # noqa: E402
from src.ingest.aciklama import aranabilir_metin, marka_kaynagi  # noqa: E402


def main() -> int:
    kip = sys.argv[1] if len(sys.argv) > 1 else "vlm"
    if kip not in {"vlm", "elle"}:
        print("[!] Kip 'vlm' veya 'elle' olmalı")
        return 1

    oznitelik_yolu = config.VERI_DIZINI / "oznitelikler.jsonl"
    urun_yolu = config.EVAL_DIZINI / "urunler.jsonl"
    for y in (oznitelik_yolu, urun_yolu):
        if not y.exists():
            print(f"[!] Dosya yok: {y}")
            return 1

    oznitelikler = [
        json.loads(s) for s in oznitelik_yolu.read_text(encoding="utf-8").splitlines() if s.strip()
    ]
    urunler = {
        json.loads(s)["dosya"]: json.loads(s)
        for s in urun_yolu.read_text(encoding="utf-8").splitlines() if s.strip()
    }

    print(f"[i] {len(oznitelikler)} ürün, kip: {kip}")

    simdi = datetime.now(timezone.utc).isoformat()
    kayitlar: list[dict] = []
    elle_doldurulan = 0

    for oz in oznitelikler:
        dosya = oz["dosya"]
        foto = config.FOTO_DIZINI / dosya
        if not foto.exists():
            continue
        urun = urunler.get(dosya, {})

        kaynak = marka_kaynagi(oz)
        marka = oz.get("marka") if kaynak == "vlm" else None

        if kip == "elle" and kaynak == "bilinmiyor":
            # Operatör kaydı tamamlamış gibi davran.
            marka = urun.get("marka")
            kaynak = "elle"
            elle_doldurulan += 1

        kayitlar.append({
            "kimlik": chroma.urun_kimligi(foto),
            "dosya": dosya,
            "kategori": oz.get("kategori"),
            "marka": marka,
            "marka_kaynagi": kaynak,
            "renk": oz.get("renk"),
            # Bu veri setinde ürün kodu yok; alan gerçek depo senaryosu için
            # şemada duruyor ve barkod okuyucudan dolacak.
            "urun_kodu": None,
            "raf": urun.get("raf"),
            "aciklama": aranabilir_metin(oz, oz.get("renk")),
            "eklendi": simdi,
        })

    baglanti = sqlite.baglan()
    sqlite.sifirla(baglanti)
    sqlite.ekle(baglanti, kayitlar)

    markali = sum(1 for k in kayitlar if k["marka"])
    print(f"[+] {sqlite.sayim(baglanti)} kayıt -> {config.SQLITE_YOLU}")
    print(f"[i] Markası olan kayıt: {markali}/{len(kayitlar)} ({markali / len(kayitlar):.1%})")
    if kip == "elle":
        print(f"[i] Elle doldurulan: {elle_doldurulan}")

    print(f"\n=== Örnek aramalar ===")
    for sorgu in ("Lino Perros", "Puma", "el cantasi", "gunes gozlugu"):
        bulunan = sqlite.ara(baglanti, sorgu, limit=3)
        if bulunan:
            satirlar = baglanti.execute(
                f"SELECT marka, kategori FROM urun WHERE kimlik IN "
                f"({','.join('?' * len(bulunan))})", bulunan
            ).fetchall()
            ozet = " | ".join(f"{s['marka'] or '-'} / {s['kategori']}" for s in satirlar)
        else:
            ozet = "(sonuç yok)"
        print(f"  {sorgu:<18} -> {ozet}")

    baglanti.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
