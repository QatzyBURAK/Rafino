"""Stok işlemlerinin duman testi — CRUD, hareketler, FTS eşitliği.

Geçici bir veritabanı ve uydurma kimliklerle çalışır; gerçek indekslere
dokunmaz. (Uydurma kimlikler ChromaDB'de bulunmadığı için silme çağrısı orada
etkisiz kalır.)

    .venv\\Scripts\\python.exe scripts\\test_stok.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.db import sqlite, stok  # noqa: E402

gecti = basarisiz = 0


def kontrol(ad: str, kosul: bool, ayrinti: str = "") -> None:
    global gecti, basarisiz
    if kosul:
        gecti += 1
        print(f"  [ OK ] {ad}")
    else:
        basarisiz += 1
        print(f"  [HATA] {ad}  {ayrinti}")


def main() -> int:
    gecici = Path(tempfile.mkdtemp()) / "test_stok.db"
    baglanti = sqlite.baglan(gecici)

    sqlite.ekle(baglanti, [
        {"kimlik": "test001", "dosya": "a.jpg", "kategori": "el cantasi",
         "marka": "Lino Perros", "marka_kaynagi": "vlm", "renk": "lacivert",
         "raf": "A-01", "aciklama": "lacivert deri el cantasi"},
        {"kimlik": "test002", "dosya": "b.jpg", "kategori": "saat",
         "marka": None, "marka_kaynagi": "bilinmiyor", "renk": "siyah",
         "raf": "B-02", "aciklama": "siyah metal saat"},
    ])

    print("\n--- Kayıt ---")
    kontrol("2 ürün eklendi", sqlite.sayim(baglanti) == 2)
    kontrol("başlangıç adedi 1", stok.urun_getir(baglanti, "test001")["adet"] == 1)

    print("\n--- Stok hareketleri ---")
    stok.hareket_ekle(baglanti, "test001", "giris", 9, "ilk parti")
    kontrol("giriş sonrası 10 adet",
            stok.urun_getir(baglanti, "test001")["adet"] == 10)

    stok.hareket_ekle(baglanti, "test001", "cikis", 3, "sevkiyat")
    kontrol("çıkış sonrası 7 adet",
            stok.urun_getir(baglanti, "test001")["adet"] == 7)

    stok.hareket_ekle(baglanti, "test001", "duzeltme", 5, "sayım farkı")
    kontrol("düzeltme sonrası 5 adet",
            stok.urun_getir(baglanti, "test001")["adet"] == 5)

    try:
        stok.hareket_ekle(baglanti, "test001", "cikis", 99)
        kontrol("stoktan fazla çıkış reddedilmeli", False, "hata fırlatmadı")
    except stok.StokHatasi:
        kontrol("stoktan fazla çıkış reddedildi", True)

    gecmis = stok.hareketler(baglanti, "test001")
    kontrol("hareket geçmişi 3 kayıt", len(gecmis) == 3, f"{len(gecmis)} bulundu")
    kontrol("hareketler önceki/sonraki tutuyor",
            gecmis[0]["onceki"] == 1 and gecmis[0]["sonraki"] == 10)

    print("\n--- Güncelleme ve FTS eşitliği ---")
    kontrol("marka aranabiliyor (güncelleme öncesi)",
            "test002" not in sqlite.ara(baglanti, "Casio"))
    stok.guncelle(baglanti, "test002", {"marka": "Casio"})
    kontrol("marka güncellendi",
            stok.urun_getir(baglanti, "test002")["marka"] == "Casio")
    kontrol("marka kaynağı otomatik 'elle' oldu (K11)",
            stok.urun_getir(baglanti, "test002")["marka_kaynagi"] == "elle")
    kontrol("FTS güncellendi, yeni marka aranabiliyor",
            "test002" in sqlite.ara(baglanti, "Casio"))

    try:
        stok.guncelle(baglanti, "test001", {"adet": 500})
        kontrol("adet doğrudan güncellenememeli", False, "hata fırlatmadı")
    except stok.StokHatasi:
        kontrol("adet doğrudan güncellenemez (hareketle değişir)", True)

    print("\n--- Silme ---")
    kontrol("silmeden önce aramada var",
            "test001" in sqlite.ara(baglanti, "Lino Perros"))
    stok.sil(baglanti, "test001", "hasarlı")
    silinen = stok.urun_getir(baglanti, "test001")
    kontrol("kayıt duruyor ama durum 'silindi'", silinen["durum"] == "silindi")
    kontrol("adet sıfırlandı", silinen["adet"] == 0)
    kontrol("FTS'ten çıkarıldı",
            "test001" not in sqlite.ara(baglanti, "Lino Perros"))
    kontrol("silme hareketi kaydedildi",
            any(h["tip"] == "silme" for h in stok.hareketler(baglanti, "test001")))
    kontrol("geçmiş hareketler korundu",
            len(stok.hareketler(baglanti, "test001")) == 4)

    try:
        stok.hareket_ekle(baglanti, "test001", "giris", 1)
        kontrol("silinmiş ürüne hareket eklenememeli", False, "hata fırlatmadı")
    except stok.StokHatasi:
        kontrol("silinmiş ürüne hareket eklenemez", True)

    print("\n--- Özet ---")
    ozet = stok.stok_ozeti(baglanti)
    kontrol("aktif çeşit 1", ozet["cesit"] == 1, str(ozet))
    kontrol("silinen 1", ozet["silinen"] == 1, str(ozet))
    kontrol("markası eksik 0 (Casio girildi)", ozet["markasi_eksik"] == 0, str(ozet))

    baglanti.close()
    gecici.unlink(missing_ok=True)

    print(f"\n{'=' * 46}")
    print(f"  geçen: {gecti}   başarısız: {basarisiz}")
    return 0 if basarisiz == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
