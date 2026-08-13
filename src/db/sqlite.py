"""SQLite + FTS5 — İndeks C, anahtar kelime araması.

Neden vektör aramasının yanına bir de bu geliyor:
Embedding modelleri nadir belirteçlerde kötü. "Samsonite" ile "Samsung" vektör
uzayında yakın çıkabiliyor, "SM-A546B" gibi ürün kodları ise hiç öğrenilmemiş
oluyor. Klasik anahtar kelime araması bunları tam eşleşmeyle çözüyor.

Ölçüm bunu doğruladı: metin indeksi (B) marka sorgularında görsel indeksten
bile kötü (R@5 0.419 / 0.585), çünkü VLM markayı ürünlerin ancak %43'ünde
okuyabiliyor ve açıklamaların çoğunda marka yok.

Türkçe notu: FTS5'in `unicode61` belirteçleyicisi `remove_diacritics 2` ile
ş→s, ğ→g, ı→i dönüşümü yapıyor. Bu, açıklamaların ASCII (`kirmizi gomlek`),
sorguların düzgün Türkçe (`kırmızı gömlek`) yazıldığı durumda da eşleşme
sağlıyor. Çekim eki sorununu (valiz / valizler) çözmüyor; onun için önek
araması kullanılıyor.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from src import config

SEMA = """
CREATE TABLE IF NOT EXISTS urun (
    kimlik        TEXT PRIMARY KEY,
    dosya         TEXT NOT NULL,
    kategori      TEXT,
    marka         TEXT,
    marka_kaynagi TEXT,
    renk          TEXT,
    urun_kodu     TEXT,
    raf           TEXT,
    aciklama      TEXT,
    adet          INTEGER NOT NULL DEFAULT 1,
    durum         TEXT NOT NULL DEFAULT 'aktif',
    eklendi       TEXT,
    guncellendi   TEXT
);

-- Stok hareketleri yalnızca EKLENİR, hiç güncellenmez veya silinmez.
-- Bir stok sisteminde "şu an kaç tane var" sorusundan daha önemli soru
-- "neden bu kadar" sorusudur; geçmiş silinirse o cevap kaybolur.
CREATE TABLE IF NOT EXISTS hareket (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    kimlik    TEXT NOT NULL,
    tip       TEXT NOT NULL,          -- giris | cikis | duzeltme | silme
    miktar    INTEGER NOT NULL,
    onceki    INTEGER,
    sonraki   INTEGER,
    aciklama  TEXT,
    kullanici TEXT,
    tarih     TEXT NOT NULL,
    FOREIGN KEY (kimlik) REFERENCES urun (kimlik)
);

CREATE INDEX IF NOT EXISTS hareket_kimlik ON hareket (kimlik);
CREATE INDEX IF NOT EXISTS urun_durum ON urun (durum);

CREATE VIRTUAL TABLE IF NOT EXISTS urun_fts USING fts5(
    kimlik UNINDEXED,
    marka,
    kategori,
    urun_kodu,
    aciklama,
    tokenize = "unicode61 remove_diacritics 2"
);
"""


def baglan(yol: Path | None = None) -> sqlite3.Connection:
    yol = yol or config.SQLITE_YOLU
    yol.parent.mkdir(parents=True, exist_ok=True)
    baglanti = sqlite3.connect(yol)
    baglanti.row_factory = sqlite3.Row
    baglanti.executescript(SEMA)
    return baglanti


def sifirla(baglanti: sqlite3.Connection) -> None:
    baglanti.executescript("DELETE FROM urun_fts; DELETE FROM urun;")
    baglanti.commit()


def ekle(baglanti: sqlite3.Connection, kayitlar: list[dict]) -> int:
    """Ürünleri hem ilişkisel tabloya hem FTS tablosuna yazar."""
    for k in kayitlar:
        baglanti.execute(
            """INSERT OR REPLACE INTO urun
               (kimlik, dosya, kategori, marka, marka_kaynagi, renk,
                urun_kodu, raf, aciklama, eklendi)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (k["kimlik"], k["dosya"], k.get("kategori"), k.get("marka"),
             k.get("marka_kaynagi"), k.get("renk"), k.get("urun_kodu"),
             k.get("raf"), k.get("aciklama"), k.get("eklendi")),
        )
        baglanti.execute("DELETE FROM urun_fts WHERE kimlik = ?", (k["kimlik"],))
        baglanti.execute(
            """INSERT INTO urun_fts (kimlik, marka, kategori, urun_kodu, aciklama)
               VALUES (?, ?, ?, ?, ?)""",
            (k["kimlik"], k.get("marka") or "", k.get("kategori") or "",
             k.get("urun_kodu") or "", k.get("aciklama") or ""),
        )
    baglanti.commit()
    return len(kayitlar)


def _sorgu_hazirla(sorgu: str) -> str:
    """Kullanıcı metnini güvenli bir FTS5 MATCH ifadesine çevirir.

    FTS5'in kendi sorgu dili var (AND, OR, NOT, tırnak, yıldız, parantez).
    Kullanıcının yazdığını doğrudan geçirmek hem sözdizimi hatası hem beklenmedik
    davranış üretir. Kelimeler ayrıştırılıp her biri tırnak içine alınıyor ve
    sonuna `*` konuyor.

    `*` önek eşleşmesi sağlıyor ve Türkçe çekim eklerini kısmen kurtarıyor:
    "valiz*" ifadesi "valizler" ve "valizi" kayıtlarını da buluyor. Tersi
    geçerli değil — "valizler" araması "valiz" kaydını bulmuyor; tam çözüm
    için gövdeleyici gerekir, bu kapsamda yok.
    """
    kelimeler = re.findall(r"\w+", sorgu, flags=re.UNICODE)
    if not kelimeler:
        return ""
    return " OR ".join(f'"{k}"*' for k in kelimeler)


def ara(baglanti: sqlite3.Connection, sorgu: str, limit: int = 20) -> list[str]:
    """BM25 ile sıralanmış kimlik listesi döner.

    Sütun ağırlıkları: marka en önemli, sonra ürün kodu. Açıklama düşük ağırlıklı
    çünkü zaten İndeks B tarafından anlamsal olarak aranıyor; burada tekrar
    öne çıkması hibriti tek yöne eğiyor.
    """
    ifade = _sorgu_hazirla(sorgu)
    if not ifade:
        return []
    try:
        satirlar = baglanti.execute(
            """SELECT kimlik
               FROM urun_fts
               WHERE urun_fts MATCH ?
               ORDER BY bm25(urun_fts, 0.0, 10.0, 3.0, 8.0, 1.0)
               LIMIT ?""",
            (ifade, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Bozuk sorgu ifadesi aramayı çökertmemeli; boş sonuç dönmek yeterli.
        return []
    return [s["kimlik"] for s in satirlar]


def sayim(baglanti: sqlite3.Connection) -> int:
    return baglanti.execute("SELECT COUNT(*) FROM urun").fetchone()[0]
