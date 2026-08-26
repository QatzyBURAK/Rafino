"""Kapalı sözlükteki ASCII yazımları düzgün Türkçeye çevirir.

Neden gerekti:
VLM istemindeki kelime listeleri (kategori, renk, malzeme, örnek ifadeler)
ASCII yazılmıştı. Model listeden kopyalarken harfleri de aynen kopyaladığı için
`gomlek`, `kirmizi`, `el cantasi` kaydedildi. Serbest metin üretirken ise model
zaten düzgün Türkçe yazıyordu; sonuç aynı cümlede karışık oldu:

    "yesil cam parfum. ... koyu yeşil içeriği."
     ^^^^^ listeden                ^^^^^ serbest metin

İstem düzeltildi (bkz. prompts/oznitelik_renkli.txt), ama eldeki kayıtlar eski
yazımda kaldı. Bu betik onları çevirir.

Yalnızca KAPALI SÖZLÜK çevriliyor: kategori ve renk listeleri sabit, karşılıkları
belirsizlik içermiyor. Serbest metindeki her kelimeyi tahminle düzeltmeye
kalkmıyoruz — orada yanlış düzeltme, eksik düzeltmeden kötü.

Arama: bu betik yazıldığında "FTS5 zaten her iki yazımı eşitliyor" sanılıyordu.
YANLIŞTI — `remove_diacritics` ş/ğ/ü/ö/ç'yi katlıyor ama `ı`yı katlamıyor, çünkü
`ı` (U+0131) aksanlı bir harf değil, başlı başına bir taban harf. Kayıtlar düzgün
Türkçeye çevrilince "kirmizi" araması `kırmızı` kaydını bulamaz hâle geldi.

Gerçek çözüm `sqlite.tr_katla()`: hem indekse yazılan metin hem sorgu aynı
katlamadan geçiyor. Bu betik de FTS satırlarını `sqlite.fts_yaz()` üzerinden
yazıyor, yani katlama otomatik uygulanıyor.

Kullanım:
    python scripts/turkce_duzelt.py            # kuru çalıştırma, hiçbir şey yazmaz
    python scripts/turkce_duzelt.py --uygula   # veritabanını günceller
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.db import sqlite

# Kapalı sözlük: istemde listelenmiş, karşılığı tek olan terimler.
# Uzun ifadeler önce gelmeli ("el cantasi" -> "el çantası"), yoksa "cantasi"
# tek başına eşleşip "el cantasi" bozuk kalır.
KARSILIKLAR: list[tuple[str, str]] = [
    # kategoriler (uzundan kısaya)
    ("gunluk ayakkabi", "günlük ayakkabı"),
    ("spor ayakkabi", "spor ayakkabı"),
    ("gunes gozlugu", "güneş gözlüğü"),
    ("sirt cantasi", "sırt çantası"),
    ("kisa pantolon", "kısa pantolon"),
    ("el cantasi", "el çantası"),
    ("ayakkabi", "ayakkabı"),
    ("gozlugu", "gözlüğü"),
    ("cantasi", "çantası"),
    ("tisort", "tişört"),
    ("gomlek", "gömlek"),
    ("cuzdan", "cüzdan"),
    ("parfum", "parfüm"),
    ("sapka", "şapka"),
    # renkler
    ("kirmizi", "kırmızı"),
    ("sari", "sarı"),
    ("yesil", "yeşil"),
    # malzemeler
    ("sentetik deri", "sentetik deri"),
    ("kumas", "kumaş"),
    ("ahsap", "ahşap"),
    ("kagit", "kağıt"),
    ("kaucuk", "kauçuk"),
    # durum
    ("saglam", "sağlam"),
    ("hasarli", "hasarlı"),
]

# Sözcük sınırı: "sari" -> "sarı" olurken "sarim" bozulmasın.
_KURALLAR = [
    (re.compile(rf"(?<![0-9A-Za-zçğıöşüÇĞİÖŞÜ]){re.escape(a)}(?![0-9A-Za-zçğıöşüÇĞİÖŞÜ])"), b)
    for a, b in KARSILIKLAR
]


def cevir(metin: str | None) -> str | None:
    """Kapalı sözlükteki ASCII yazımları Türkçeye çevirir."""
    if not metin:
        return metin
    for kalip, karsilik in _KURALLAR:
        metin = kalip.sub(karsilik, metin)
    return metin


def main() -> None:
    # Windows konsolu varsayılan olarak cp1254; Türkçe harfler ve "→" gibi
    # işaretler UnicodeEncodeError veriyor. Çıktıyı UTF-8'e alıyoruz ki
    # betiğin kendisi de tam olarak düzeltmeye çalıştığı hataya düşmesin.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    uygula = "--uygula" in sys.argv
    baglanti = sqlite.baglan()

    degisenler = []
    for satir in baglanti.execute(
        "SELECT kimlik, kategori, renk, aciklama FROM urun"
    ).fetchall():
        d = dict(satir)
        yeni = {alan: cevir(d[alan]) for alan in ("kategori", "renk", "aciklama")}
        farklar = {a: (d[a], yeni[a]) for a in yeni if d[a] != yeni[a]}
        if farklar:
            degisenler.append((d["kimlik"], farklar))

    print(f"[i] {len(degisenler)} kayıtta değişiklik var\n")
    for kimlik, farklar in degisenler[:12]:
        print(f"  {kimlik[:12]}")
        for alan, (eski, yeni) in farklar.items():
            print(f"    {alan}: {eski!r}")
            print(f"    {' ' * len(alan)}→ {yeni!r}")
    if len(degisenler) > 12:
        print(f"  ... ve {len(degisenler) - 12} kayıt daha")

    if not uygula:
        print("\n[i] Kuru çalıştırma. Uygulamak için: --uygula")
        return

    if not degisenler:
        print("[i] Değişecek bir şey yok.")
        return

    # Yedek: bu betik geri alma sunmuyor, o yüzden dosyayı kopyalıyoruz.
    damga = datetime.now().strftime("%Y%m%d-%H%M%S")
    yedek = config.SQLITE_YOLU.with_suffix(f".{damga}.yedek.db")
    shutil.copy2(config.SQLITE_YOLU, yedek)
    print(f"\n[+] Yedek: {yedek.name}")

    for kimlik, farklar in degisenler:
        baglanti.execute(
            "UPDATE urun SET "
            + ", ".join(f"{alan} = ?" for alan in farklar)
            + " WHERE kimlik = ?",
            [yeni for _, yeni in farklar.values()] + [kimlik],
        )
        # FTS satırı elle eşitleniyor. `urun_fts` üzerinde tetikleyici YOK;
        # normalde bu eşitlemeyi `stok.guncelle` yapıyor ama burada ham UPDATE
        # kullanıyoruz. Atlanırsa indeks C eski metinle kalır ve `ı` içeren
        # sorgular kayar (bkz. modül başlığı).
        satir = baglanti.execute(
            "SELECT marka, kategori, urun_kodu, aciklama FROM urun WHERE kimlik = ?",
            (kimlik,),
        ).fetchone()
        sqlite.fts_yaz(baglanti, kimlik, satir["marka"], satir["kategori"],
                       satir["urun_kodu"], satir["aciklama"])
    baglanti.commit()
    print(f"[+] {len(degisenler)} kayıt güncellendi (FTS dahil).")
    print("[!] Açıklamalar değiştiyse İndeks B yeniden kurulmalı:")
    print("    python scripts/indeks_b_kur.py")


if __name__ == "__main__":
    main()
