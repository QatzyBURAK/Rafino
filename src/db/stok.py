"""Stok işlemleri — güncelleme, silme, giriş/çıkış hareketleri.

Şirketin özellik listesinde eksik olan kısım buydu: sistem şimdiye kadar sadece
kayıt ekleyip arayabiliyordu.

İki tasarım kuralı:

1. **Hareket kaydı silinmez.** `hareket` tablosuna yalnızca ekleme yapılır.
   Bir stok sisteminde "şu an kaç tane var" sorusundan daha kritik soru "neden
   bu kadar" sorusudur; hareket geçmişi silinirse sayım farkı hiç açıklanamaz.

2. **Ürün silme, kaydı yok etmez.** `durum = 'silindi'` yapılır. Böylece geçmiş
   hareketler öksüz kalmaz. Ancak arama indekslerinden GERÇEKTEN çıkarılır —
   yoksa silinen ürün aramada çıkmaya devam eder ve bu sessiz bir hata olur.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from src.db import chroma, sqlite


def _simdi() -> str:
    return datetime.now(timezone.utc).isoformat()


class StokHatasi(Exception):
    """İşlem, stok kurallarına aykırı olduğu için reddedildi."""


def urun_getir(baglanti: sqlite3.Connection, kimlik: str) -> dict | None:
    satir = baglanti.execute(
        "SELECT * FROM urun WHERE kimlik = ?", (kimlik,)
    ).fetchone()
    return dict(satir) if satir else None


def guncelle(
    baglanti: sqlite3.Connection,
    kimlik: str,
    alanlar: dict,
    kullanici: str = "sistem",
) -> dict:
    """Ürün alanlarını günceller ve FTS indeksini eşitler.

    FTS satırının da güncellenmesi şart: sadece `urun` tablosu değişirse arama
    eski marka/kategoriyle sonuç vermeye devam eder ve bu fark edilmez.
    """
    mevcut = urun_getir(baglanti, kimlik)
    if mevcut is None:
        raise StokHatasi(f"Ürün bulunamadı: {kimlik}")
    if mevcut["durum"] != "aktif":
        raise StokHatasi(f"Ürün silinmiş durumda: {kimlik}")

    izinli = {"kategori", "marka", "marka_kaynagi", "renk",
              "urun_kodu", "raf", "aciklama"}
    gecersiz = set(alanlar) - izinli
    if gecersiz:
        raise StokHatasi(f"Güncellenemeyecek alanlar: {sorted(gecersiz)}")
    if not alanlar:
        return mevcut

    # Marka elle düzeltildiyse kaynağı da işaretle (K11).
    if "marka" in alanlar and "marka_kaynagi" not in alanlar:
        alanlar = {**alanlar, "marka_kaynagi": "elle"}

    atama = ", ".join(f"{a} = ?" for a in alanlar)
    baglanti.execute(
        f"UPDATE urun SET {atama}, guncellendi = ? WHERE kimlik = ?",
        (*alanlar.values(), _simdi(), kimlik),
    )

    yeni = urun_getir(baglanti, kimlik)
    sqlite.fts_yaz(baglanti, kimlik, yeni.get("marka"), yeni.get("kategori"),
                   yeni.get("urun_kodu"), yeni.get("aciklama"))
    baglanti.commit()
    return yeni


def hareket_ekle(
    baglanti: sqlite3.Connection,
    kimlik: str,
    tip: str,
    miktar: int,
    aciklama: str = "",
    kullanici: str = "sistem",
) -> dict:
    """Stok giriş/çıkış/düzeltme hareketi işler.

    tip:
      giris     stoğa ekleme (miktar pozitif)
      cikis     stoktan çıkarma (miktar pozitif verilir, adetten düşülür)
      duzeltme  sayım farkı; miktar YENİ adet olarak yorumlanır
    """
    if tip not in {"giris", "cikis", "duzeltme"}:
        raise StokHatasi(f"Geçersiz hareket tipi: {tip}")
    if miktar < 0:
        raise StokHatasi("Miktar negatif olamaz")

    urun = urun_getir(baglanti, kimlik)
    if urun is None:
        raise StokHatasi(f"Ürün bulunamadı: {kimlik}")
    if urun["durum"] != "aktif":
        raise StokHatasi(f"Ürün silinmiş durumda: {kimlik}")

    onceki = urun["adet"]
    if tip == "giris":
        sonraki = onceki + miktar
    elif tip == "cikis":
        if miktar > onceki:
            # Sessizce sıfıra çekmiyoruz: negatife düşen çıkış, ya sayım hatası
            # ya yanlış ürün demektir ve operatörün bunu görmesi gerekir.
            raise StokHatasi(
                f"Stokta {onceki} adet var, {miktar} adet çıkış yapılamaz"
            )
        sonraki = onceki - miktar
    else:
        sonraki = miktar

    baglanti.execute(
        "UPDATE urun SET adet = ?, guncellendi = ? WHERE kimlik = ?",
        (sonraki, _simdi(), kimlik),
    )
    baglanti.execute(
        """INSERT INTO hareket
           (kimlik, tip, miktar, onceki, sonraki, aciklama, kullanici, tarih)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (kimlik, tip, miktar, onceki, sonraki, aciklama, kullanici, _simdi()),
    )
    baglanti.commit()
    return {"kimlik": kimlik, "tip": tip, "onceki": onceki, "sonraki": sonraki}


def sil(
    baglanti: sqlite3.Connection,
    kimlik: str,
    aciklama: str = "",
    kullanici: str = "sistem",
) -> dict:
    """Ürünü stoktan çıkarır.

    Kayıt veritabanında kalır (`durum = 'silindi'`) ama arama indekslerinden
    tamamen çıkarılır. İkisi birlikte yapılmazsa silinen ürün aramada çıkmaya
    devam eder ve kimse fark etmez.
    """
    urun = urun_getir(baglanti, kimlik)
    if urun is None:
        raise StokHatasi(f"Ürün bulunamadı: {kimlik}")
    if urun["durum"] == "silindi":
        return urun

    onceki = urun["adet"]
    baglanti.execute(
        "UPDATE urun SET durum = 'silindi', adet = 0, guncellendi = ? WHERE kimlik = ?",
        (_simdi(), kimlik),
    )
    baglanti.execute(
        """INSERT INTO hareket
           (kimlik, tip, miktar, onceki, sonraki, aciklama, kullanici, tarih)
           VALUES (?, 'silme', ?, ?, 0, ?, ?, ?)""",
        (kimlik, onceki, onceki, aciklama, kullanici, _simdi()),
    )
    baglanti.execute("DELETE FROM urun_fts WHERE kimlik = ?", (kimlik,))
    baglanti.commit()

    # Vektör indekslerinden de çıkar.
    for koleksiyon in (chroma.gorsel_koleksiyon(), chroma.metin_koleksiyon()):
        try:
            koleksiyon.delete(ids=[kimlik])
        except Exception:  # noqa: BLE001 - kayıt yoksa sorun değil
            pass

    return urun_getir(baglanti, kimlik)


def hareketler(baglanti: sqlite3.Connection, kimlik: str) -> list[dict]:
    """Bir ürünün tüm hareket geçmişi, eskiden yeniye."""
    return [
        dict(s) for s in baglanti.execute(
            "SELECT * FROM hareket WHERE kimlik = ? ORDER BY id", (kimlik,)
        )
    ]


def stok_ozeti(baglanti: sqlite3.Connection) -> dict:
    satir = baglanti.execute(
        """SELECT COUNT(*) AS cesit,
                  COALESCE(SUM(adet), 0) AS toplam_adet,
                  SUM(CASE WHEN adet = 0 THEN 1 ELSE 0 END) AS tukenen
           FROM urun WHERE durum = 'aktif'"""
    ).fetchone()
    silinen = baglanti.execute(
        "SELECT COUNT(*) FROM urun WHERE durum = 'silindi'"
    ).fetchone()[0]
    markasiz = baglanti.execute(
        """SELECT COUNT(*) FROM urun
           WHERE durum = 'aktif' AND (marka IS NULL OR marka = '')"""
    ).fetchone()[0]
    return {
        "cesit": satir["cesit"],
        "toplam_adet": satir["toplam_adet"],
        "tukenen": satir["tukenen"],
        "silinen": silinen,
        # K11: markası bilinmeyen kayıtlar operatöre gösterilecek liste.
        "markasi_eksik": markasiz,
    }
