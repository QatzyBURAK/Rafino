"""SQLite + FTS5 — İndeks C, anahtar kelime araması.

Neden vektör aramasının yanına bir de bu geliyor:
Embedding modelleri nadir belirteçlerde kötü. "Samsonite" ile "Samsung" vektör
uzayında yakın çıkabiliyor, "SM-A546B" gibi ürün kodları ise hiç öğrenilmemiş
oluyor. Klasik anahtar kelime araması bunları tam eşleşmeyle çözüyor.

Ölçüm bunu doğruladı: metin indeksi (B) marka sorgularında görsel indeksten
bile kötü (R@5 0.419 / 0.585), çünkü VLM markayı ürünlerin ancak %43'ünde
okuyabiliyor ve açıklamaların çoğunda marka yok.

Türkçe notu: FTS5'in `unicode61 remove_diacritics 2` belirteçleyicisi ş→s, ğ→g,
ü→u, ö→o, ç→c katlamasını yapıyor ama **ı→i yapmıyor**. Sebebi Unicode: ş, ğ, ü,
ö, ç harflerinin hepsi "taban harf + birleşen işaret" olarak ayrışıyor ve
işaretin atılması onları ASCII'ye indiriyor. `ı` (U+0131, DOTLESS I) ise başlı
başına bir taban harf — atılacak bir aksanı yok, olduğu gibi kalıyor.

Sonuç: Türkçe klavyesi olmayan bir operatör "kirmizi" yazdığında `kırmızı`
kaydını bulamıyordu. Üstelik `ı`, tam da bu alanın sözlüğünde en sık geçen harf:
kırmızı, sarı, ayakkabı, çantası, sırt, kısa.

Çözüm: hem indekse yazılan metin hem de sorgu `tr_katla()` ile ASCII'ye
indirgeniyor. FTS bir görüntüleme yüzeyi değil, eşleştirme yapısı; iki tarafı
aynı biçime getirmek doğru olanı. Gösterilen metin `urun` tablosunda düzgün
Türkçe kalıyor.

Çekim eki sorununu (valiz / valizler) bu çözmüyor; onun için önek araması var.
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

CREATE VIRTUAL TABLE IF NOT EXISTS urun_fts USING fts5(
    kimlik UNINDEXED,
    marka,
    kategori,
    urun_kodu,
    aciklama,
    tokenize = "unicode61 remove_diacritics 2"
);
"""


# Şema büyüdükçe eklenen sütunlar. `CREATE TABLE IF NOT EXISTS` var olan bir
# tabloya sütun EKLEMEZ; eski bir veritabanı açıldığında bu sütunlar eksik kalır
# ve sorgular "no such column" ile patlar. Gerçek bir stok sisteminde veritabanı
# silinip yeniden kurulamayacağı için eksikler ALTER TABLE ile tamamlanıyor.
EK_SUTUNLAR = {
    "adet": "INTEGER NOT NULL DEFAULT 1",
    "durum": "TEXT NOT NULL DEFAULT 'aktif'",
    "guncellendi": "TEXT",
}

INDEKSLER = """
CREATE INDEX IF NOT EXISTS hareket_kimlik ON hareket (kimlik);
CREATE INDEX IF NOT EXISTS urun_durum ON urun (durum);
"""


def _goc_uygula(baglanti: sqlite3.Connection) -> list[str]:
    """Eksik sütunları ekler. Eklenenlerin adını döner."""
    mevcut = {s["name"] for s in baglanti.execute("PRAGMA table_info(urun)")}
    eklenen = []
    for ad, tanim in EK_SUTUNLAR.items():
        if ad not in mevcut:
            baglanti.execute(f"ALTER TABLE urun ADD COLUMN {ad} {tanim}")
            eklenen.append(ad)
    if eklenen:
        baglanti.commit()
    return eklenen


def baglan(yol: Path | None = None) -> sqlite3.Connection:
    yol = yol or config.SQLITE_YOLU
    yol.parent.mkdir(parents=True, exist_ok=True)
    baglanti = sqlite3.connect(yol)
    baglanti.row_factory = sqlite3.Row
    baglanti.executescript(SEMA)
    # Göç, indekslerden ÖNCE: urun_durum indeksi `durum` sütununa bağlı.
    _goc_uygula(baglanti)
    baglanti.executescript(INDEKSLER)
    return baglanti


# `İ` küçük harf `i`ye iniyor: unicode61 zaten küçültme yapıyor, biz yalnızca
# harfin kimliğini sadeleştiriyoruz.
_TR_KATLAMA = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iisSgGuUoOcC")


def tr_katla(metin: str | None) -> str:
    """Türkçe harfleri ASCII karşılıklarına indirger.

    Hem FTS'e yazarken hem sorgularken çağrılıyor. İki taraf da aynı katlamadan
    geçtiği için "kırmızı", "kirmizi" ve "KIRMIZI" aynı belirteci üretiyor.

    Belirteçleyicinin kendi katlaması ş/ğ/ü/ö/ç için zaten çalışıyor; burada
    tekrar uygulanması zararsız. Asıl kazanç `ı` — bkz. modül başlığı.
    """
    return (metin or "").translate(_TR_KATLAMA)


def fts_yaz(
    baglanti: sqlite3.Connection,
    kimlik: str,
    marka: str | None,
    kategori: str | None,
    urun_kodu: str | None,
    aciklama: str | None,
) -> None:
    """Bir ürünün FTS satırını siler ve katlanmış hâliyle yeniden yazar.

    Tek giriş noktası olarak duruyor: `urun_fts` üzerinde tetikleyici yok, yani
    FTS'i güncel tutmak çağıranın sorumluluğunda. Katlamayı her çağırana
    bırakmak, bir yerde unutulduğunda sessiz arama kaybına yol açıyordu.
    """
    baglanti.execute("DELETE FROM urun_fts WHERE kimlik = ?", (kimlik,))
    baglanti.execute(
        """INSERT INTO urun_fts (kimlik, marka, kategori, urun_kodu, aciklama)
           VALUES (?, ?, ?, ?, ?)""",
        (kimlik, tr_katla(marka), tr_katla(kategori),
         tr_katla(urun_kodu), tr_katla(aciklama)),
    )


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
        fts_yaz(baglanti, k["kimlik"], k.get("marka"), k.get("kategori"),
                k.get("urun_kodu"), k.get("aciklama"))
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
    # Sorgu da indekse yazılan metinle aynı katlamadan geçiyor; aksi hâlde
    # "kirmizi" yazan operatör "kırmızı" kaydını bulamaz.
    kelimeler = re.findall(r"\w+", tr_katla(sorgu), flags=re.UNICODE)
    if not kelimeler:
        return ""
    # Kelimeler AND ile bağlanıyor, OR ile değil. OR kullanıldığında tek bir
    # kelimenin öneki tutunca alakasız kayıtlar geliyordu: "kahve makinesi"
    # sorgusunda `"kahve"*` ifadesi "kahverengi"ye eşleşip depoda kahve
    # makinesi yokken sekiz güneş gözlüğü döndürüyordu.
    #
    # İndeks C'nin işi kesinlik (marka, ürün kodu, tam kelime); geri çağırmayı
    # A ve B indeksleri sağlıyor. Bu yüzden burada dar olmak doğru: eşleşme
    # varsa güvenilir olsun, yoksa sessiz kalsın.
    return " AND ".join(f'"{k}"*' for k in kelimeler)


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
