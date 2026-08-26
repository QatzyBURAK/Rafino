"""FastAPI uygulaması — arayüzün konuştuğu uçlar.

Tasarım kuralları (hepsi ölçümle veya acıyla öğrenildi):

1. **Embedder'lar başlangıçta bir kez yüklenir.** Görsel embedder ~15 saniyede
   yükleniyor; her aramada yeniden yüklemek arayüzü kullanılamaz yapar.
   `lifespan` içinde açılıp süreç boyunca açık tutuluyor.

2. **`--reload` ile çalıştırılmaz.** Uvicorn'un yeniden yükleme kipi alt
   süreçleri öldürüyor ve kayıt hattı sessizce yarıda kalıyor.

3. **Uzun işler için "bekle bitmiştir" yok.** Fotoğraf işleme VLM gerektiriyor
   ve saniyeler sürüyor; iş kimliği veriliyor, arayüz durumu soruyor.

4. **Arka plan hataları yutulmaz.** İş kaydında hata metni tutuluyor ve
   arayüze aynen gösteriliyor.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config
from src.db import chroma, sqlite, stok
from src.ingest.aciklama import aciklama_uret
from src.ingest.vlm_servis import VlmHatasi, VlmServisi
from src.search.arama import Arayici

# Süreç boyunca açık kalan kaynaklar.
_durum: dict = {"arayici": None, "baglanti": None}

# İş parçacığı başına SQLite bağlantısı. Bkz. `_baglanti()`.
_yerel = threading.local()

# Sıcak VLM işçisi. Açılışta BAŞLATILMIYOR: modeli yüklemek 69-110 sn sürüyor ve
# VRAM tutuyor. İlk öznitelik isteğinde doğuyor, boşta kalınca kendini kapatıyor.
_vlm = VlmServisi()

# Uzun işlerin durumu. Üretimde kalıcı bir kuyruk gerekir; tek kullanıcılı
# depo arayüzü için süreç içi sözlük yeterli ve karmaşıklık eklemiyor.
_isler: dict[str, dict] = {}

# Aynı anda kaç VLM işi kabul edilir. VLM işçisinin kendi kilidi var, yani işler
# zaten sıraya giriyor; bu sınır kuyruğun sınırsız büyümesini engelliyor. Rate
# limiting bilinçli olarak eklenmedi: uygulama 127.0.0.1'e bağlı, tek kullanıcılı
# yerel bir araç ve dışarıdan erişilmiyor. Genel bir HTTP rate limiter buraya
# yanlış bir güvenlik hissi katardı; asıl korunması gereken tek pahalı kaynak
# GPU ve onu bu sayaç koruyor. Dağıtım senaryosu değişirse (çok kullanıcı, ağa
# açık) gerçek bir rate limiter ve kimlik doğrulama şart olur — bkz. NOTLAR.md.
_AZAMI_ESZAMANLI_IS = 4
_aktif_is_sayaci = 0
_is_kilidi = threading.Lock()


@asynccontextmanager
async def yasam_dongusu(app: FastAPI):
    config.dizinleri_hazirla()
    # Açılışta bir bağlantı kurulup şema ve göçler burada çalışıyor: veritabanı
    # bozuksa ilk istekte değil, sunucu açılırken hata versin. İsteklerin
    # kullandığı bağlantılar ayrı (bkz. `_baglanti()`).
    _durum["baglanti"] = sqlite.baglan()
    print("[i] Modeller yükleniyor (ilk açılışta biraz sürer)...")
    _durum["arayici"] = Arayici()
    # İlk sorguyu şimdi yapıp modelleri belleğe çekiyoruz; yoksa ilk kullanıcı
    # aramasının bedelini kullanıcı ödüyor.
    try:
        _durum["arayici"].ara("hazirlik", k=1)
        print("[+] Arama hazır.")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Isınma sorgusu başarısız (arama yine de denenecek): {exc}")
    yield
    # VLM işçisi ayrı bir süreç; kapatılmazsa sunucu inse bile VRAM'i tutmaya
    # devam eder ve ancak elle öldürülür.
    _vlm.kapat()
    if _durum["arayici"]:
        _durum["arayici"].bosalt()
    if _durum["baglanti"]:
        _durum["baglanti"].close()


# Etkileşimli API dokümanları (/docs, /redoc, /openapi.json) yalnızca geliştirmede
# açık. Üretimde bütün uç envanterini, gövde şemalarını ve örnek değerleri
# isteyen herkese sunmak saldırı yüzeyini gereksiz genişletiyor. RAFINO_URETIM
# ayarlıysa kapatılıyor. Varsayılan geliştirme çünkü tek kullanıcılı yerel bir
# araç; kapatma bilinçli bir dağıtım kararı olmalı.
_uretim = os.environ.get("RAFINO_URETIM", "").lower() in {"1", "true", "evet"}

app = FastAPI(
    title="Rafino",
    description="Görsel tabanlı akıllı stok takip sistemi",
    version="0.1.0",
    lifespan=yasam_dongusu,
    docs_url=None if _uretim else "/docs",
    redoc_url=None if _uretim else "/redoc",
    openapi_url=None if _uretim else "/openapi.json",
)


def _baglanti():
    """Bu iş parçacığına ait SQLite bağlantısını döner.

    FastAPI, eşzamansız olmayan uçları bir iş parçacığı havuzunda çalıştırıyor
    ve her istek farklı bir parçacığa düşebiliyor. SQLite bağlantıları
    parçacıklar arasında paylaşılamaz:

        sqlite3.ProgrammingError: SQLite objects created in a thread can only
        be used in that same thread.

    Bu yüzden tek bir paylaşılan bağlantı yerine parçacık başına bir bağlantı
    tutuluyor. `check_same_thread=False` + kilit de bir seçenekti; parçacık
    başına bağlantı hem kilitsiz hem daha basit, ve SQLite dosya düzeyinde
    kilitlemeyi kendisi yapıyor.
    """
    baglanti = getattr(_yerel, "baglanti", None)
    if baglanti is None:
        baglanti = _yerel.baglanti = sqlite.baglan()
    return baglanti


def _arayici() -> Arayici:
    if _durum["arayici"] is None:
        raise HTTPException(503, "Arama motoru hazır değil")
    return _durum["arayici"]


# --------------------------------------------------------------------------
# Modeller
# --------------------------------------------------------------------------

# Metin alanları için makul bir üst sınır. Sınırsız string, tek istekte
# megabaytlarca veri yazdırıp veritabanını şişirebiliyor; depo alanları için
# 200 karakter fazlasıyla yeterli. Marka/kategori/renk zaten kısa.
_METIN = Field(default=None, max_length=200)


class Guncelleme(BaseModel):
    kategori: str | None = _METIN
    marka: str | None = _METIN
    renk: str | None = _METIN
    urun_kodu: str | None = _METIN
    raf: str | None = _METIN


class Hareket(BaseModel):
    tip: Literal["giris", "cikis", "duzeltme"]
    # Üst sınır bir daktilo hatasına karşı: sınırsızken 10^18 gibi bir giriş
    # kabul ediliyordu ve tek yanlış tuş stok kaydını mahvediyordu. Bir milyon,
    # gerçek bir depo hareketinin çok üstünde ama makul.
    miktar: int = Field(ge=0, le=1_000_000)
    aciklama: str = Field(default="", max_length=500)
    kullanici: str = Field(default="operator", max_length=100)


# --------------------------------------------------------------------------
# Arama
# --------------------------------------------------------------------------

@app.get("/api/ara")
def ara(
    q: str = Query(..., min_length=1, description="Türkçe arama metni"),
    k: int = Query(10, ge=1, le=50),
):
    """Üç indekste arar ve RRF ile birleştirilmiş sonucu döner.

    Her sonuçta `siralar` alanı, kaydın hangi indekste kaçıncı sırada
    çıktığını gösteriyor. Arayüz bunu kullanıcıya gösteriyor: sonucun neden
    geldiği görünür olsun diye. Kapalı kutu bir arama, kullanıcının sisteme
    güvenmesini zorlaştırır.
    """
    arayici = _arayici()
    baglanti = _baglanti()

    bulunanlar = arayici.ara(q, k=k)
    if not bulunanlar:
        return {"sorgu": q, "adet": 0, "sonuclar": []}

    kimlikler = [b["kimlik"] for b in bulunanlar]
    yer = ",".join("?" * len(kimlikler))
    satirlar = {
        s["kimlik"]: dict(s)
        for s in baglanti.execute(
            f"SELECT * FROM urun WHERE kimlik IN ({yer}) AND durum = 'aktif'",
            kimlikler,
        )
    }

    sonuclar = []
    for b in bulunanlar:
        urun = satirlar.get(b["kimlik"])
        if urun is None:
            # Silinmiş ürün vektör indeksinde kalmış olabilir; gösterme.
            continue
        sonuclar.append({
            **urun,
            "skor": round(b["skor"], 5),
            "siralar": b["siralar"],
        })
    return {"sorgu": q, "adet": len(sonuclar), "sonuclar": sonuclar}


# --------------------------------------------------------------------------
# Ürün
# --------------------------------------------------------------------------

@app.get("/api/urun/{kimlik}")
def urun_getir(kimlik: str):
    urun = stok.urun_getir(_baglanti(), kimlik)
    if urun is None:
        raise HTTPException(404, "Ürün bulunamadı")
    urun["hareketler"] = stok.hareketler(_baglanti(), kimlik)
    return urun


@app.patch("/api/urun/{kimlik}")
def urun_guncelle(kimlik: str, govde: Guncelleme):
    alanlar = {a: d for a, d in govde.model_dump().items() if d is not None}
    if not alanlar:
        raise HTTPException(400, "Güncellenecek alan verilmedi")
    try:
        return stok.guncelle(_baglanti(), kimlik, alanlar)
    except stok.StokHatasi as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/urun/{kimlik}")
def urun_sil(kimlik: str, aciklama: str = ""):
    try:
        return stok.sil(_baglanti(), kimlik, aciklama)
    except stok.StokHatasi as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/urun/{kimlik}/hareket")
def hareket_ekle(kimlik: str, govde: Hareket):
    try:
        return stok.hareket_ekle(
            _baglanti(), kimlik, govde.tip, govde.miktar,
            govde.aciklama, govde.kullanici,
        )
    except stok.StokHatasi as exc:
        # 400 dönüyoruz çünkü bu bir sunucu hatası değil, iş kuralı ihlali:
        # "stokta 3 varken 5 çıkış yapılamaz" gibi. Arayüz mesajı aynen gösterir.
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/foto/{dosya}")
def foto(dosya: str):
    # Yol geçişi engelleniyor: sadece dosya adı kabul ediliyor, dizin değil.
    guvenli = Path(dosya).name
    yol = config.FOTO_DIZINI / guvenli
    if not yol.exists():
        raise HTTPException(404, "Fotoğraf bulunamadı")
    return FileResponse(yol)


# --------------------------------------------------------------------------
# Stok genel
# --------------------------------------------------------------------------

@app.get("/api/ozet")
def ozet():
    baglanti = _baglanti()
    o = stok.stok_ozeti(baglanti)
    o["gorsel_indeks"] = chroma.gorsel_koleksiyon().count()
    o["metin_indeks"] = chroma.metin_koleksiyon().count()
    return o


@app.get("/api/urunler")
def urunler(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    eksik_marka: bool = Query(False, description="Yalnızca markası bilinmeyenler"),
):
    """Ürün listesi. `eksik_marka=true` K11 kararının arayüz karşılığı:
    operatörün tamamlaması gereken kayıtlar."""
    kosul = "durum = 'aktif'"
    if eksik_marka:
        kosul += " AND (marka IS NULL OR marka = '')"
    satirlar = _baglanti().execute(
        f"SELECT * FROM urun WHERE {kosul} ORDER BY eklendi DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    toplam = _baglanti().execute(
        f"SELECT COUNT(*) FROM urun WHERE {kosul}"
    ).fetchone()[0]
    return {"toplam": toplam, "urunler": [dict(s) for s in satirlar]}


# --------------------------------------------------------------------------
# Uzun iş: fotoğraftan kayıt
# --------------------------------------------------------------------------

def _is_calistir(is_id: str, foto_yolu: Path) -> None:
    """Arka planda çalışır. Hata olursa iş kaydına yazılır — yutulmaz."""
    from src.ingest.kopya import kopya_kontrol, mesaj

    try:
        _isler[is_id]["durum"] = "kopya_kontrolu"
        bulgu = kopya_kontrol(foto_yolu, _arayici().gorsel_embedder)
        _isler[is_id]["kopya"] = {
            "durum": bulgu.durum,
            "onay_gerekli": bulgu.onay_gerekli,
            "mesaj": mesaj(bulgu),
            "eslesmeler": bulgu.eslesmeler,
        }
        if bulgu.onay_gerekli:
            # Operatöre soruluyor; kayıt onaya kadar ilerlemiyor.
            _isler[is_id]["durum"] = "onay_bekliyor"
            return
        _isler[is_id]["durum"] = "oznitelik_bekliyor"
        # VLM ayrı süreçte çalıştırılacak (scripts/vlm_toplu.py). Arayüzden
        # tek fotoğraf işlemek için o hat henüz bağlanmadı; bu uç şimdilik
        # kopya kontrolüne kadar ilerliyor ve durumu dürüstçe bildiriyor.
        _isler[is_id]["durum"] = "beklemede"
        _isler[is_id]["not"] = (
            "Kopya kontrolü tamam. Öznitelik çıkarımı henüz arayüze bağlanmadı; "
            "toplu kayıt için scripts/vlm_toplu.py kullanılıyor."
        )
    except Exception as exc:  # noqa: BLE001
        _isler[is_id]["durum"] = "hata"
        _isler[is_id]["hata"] = f"{type(exc).__name__}: {exc}"


@app.post("/api/is/kopya-kontrol")
def kopya_kontrol_baslat(dosya: str, arka_plan: BackgroundTasks):
    """Var olan bir fotoğraf için kopya kontrolü başlatır ve iş kimliği döner."""
    yol = config.FOTO_DIZINI / Path(dosya).name
    if not yol.exists():
        raise HTTPException(404, "Fotoğraf bulunamadı")
    is_id = uuid.uuid4().hex[:12]
    _isler[is_id] = {
        "id": is_id,
        "dosya": yol.name,
        "durum": "sirada",
        "baslangic": datetime.now(timezone.utc).isoformat(),
    }
    arka_plan.add_task(_is_calistir, is_id, yol)
    return {"is_id": is_id, "durum": "sirada"}


@app.get("/api/is/{is_id}")
def is_durumu(is_id: str):
    """Arayüz bu ucu yoklayarak gerçek bitişi öğrenir.

    Sabit süre bekleyip "herhalde bitmiştir" demek yanlış cevap üretiyor;
    durum buradan sorulur.
    """
    kayit = _isler.get(is_id)
    if kayit is None:
        raise HTTPException(404, "İş bulunamadı")
    return kayit


# --------------------------------------------------------------------------
# Ürün ekleme
# --------------------------------------------------------------------------

# Depo fotoğrafları telefondan geliyor; 25 MB pratikte fazlasıyla yeterli.
# Sınır olmadan tek istek belleği doldurabiliyor.
AZAMI_FOTO_BAYT = 25 * 1024 * 1024
GECERLI_UZANTILAR = {".jpg", ".jpeg", ".png", ".webp"}


@app.post("/api/foto")
async def foto_yukle(dosya: UploadFile = File(...)):
    """Fotoğrafı içerik özetiyle adlandırıp kaydeder ve kopya kontrolü yapar.

    Dosya adı olarak kullanıcının verdiği ad KULLANILMIYOR; kimlik zaten
    içerikten türetiliyor (bkz. `chroma.urun_kimligi`) ve dosya da o kimlikle
    saklanıyor. Böylece aynı fotoğraf ikinci kez yüklenirse aynı dosyaya denk
    geliyor, çakışma ve yol geçişi riski birlikte ortadan kalkıyor.
    """
    icerik = await dosya.read()
    if not icerik:
        raise HTTPException(400, "Boş dosya")
    if len(icerik) > AZAMI_FOTO_BAYT:
        mb = AZAMI_FOTO_BAYT // (1024 * 1024)
        raise HTTPException(413, f"Fotoğraf {mb} MB sınırını aşıyor")

    uzanti = Path(dosya.filename or "").suffix.lower()
    if uzanti not in GECERLI_UZANTILAR:
        raise HTTPException(
            400, f"Desteklenmeyen dosya türü. Kabul edilenler: "
                 f"{', '.join(sorted(GECERLI_UZANTILAR))}"
        )

    # Uzantıya güvenilmiyor: gerçekten çözülebilen bir görsel mi, açarak bakılıyor.
    try:
        from PIL import Image

        Image.open(io.BytesIO(icerik)).verify()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "Dosya geçerli bir görsel değil") from exc

    kimlik = hashlib.sha256(icerik).hexdigest()[:16]
    hedef = config.FOTO_DIZINI / f"{kimlik}{uzanti}"
    if not hedef.exists():
        hedef.write_bytes(icerik)

    kayitli = stok.urun_getir(_baglanti(), kimlik)
    return {
        "dosya": hedef.name,
        "kimlik": kimlik,
        "zaten_kayitli": kayitli is not None,
        "kayit": kayitli,
    }


def _is_yeri_ayir() -> None:
    """Eşzamanlı VLM iş sınırını uygular. Doluysa HTTPException(429) atar."""
    global _aktif_is_sayaci
    with _is_kilidi:
        if _aktif_is_sayaci >= _AZAMI_ESZAMANLI_IS:
            raise HTTPException(
                429, "Şu an çok fazla fotoğraf işleniyor; biraz sonra deneyin"
            )
        _aktif_is_sayaci += 1


def _is_yeri_birak() -> None:
    global _aktif_is_sayaci
    with _is_kilidi:
        _aktif_is_sayaci = max(0, _aktif_is_sayaci - 1)


def _oznitelik_isi(is_id: str, foto_yolu: Path) -> None:
    """Arka planda VLM'den öznitelik çıkarır. Hata yutulmaz, işe yazılır."""
    try:
        _isler[is_id]["durum"] = (
            "model_yukleniyor" if not _vlm.acik else "cikariliyor"
        )
        oznitelik = _vlm.oznitelik_cikar(foto_yolu)
        _isler[is_id]["oznitelik"] = oznitelik
        _isler[is_id]["durum"] = "bitti"
    except VlmHatasi as exc:
        # Operatör bilgileri elle de girebiliyor; VLM'in başarısızlığı akışı
        # durdurmuyor, yalnızca ön doldurma olmuyor. Mesaj arayüzde aynen
        # gösteriliyor ki "neden boş geldi" sorusu cevapsız kalmasın.
        _isler[is_id]["durum"] = "hata"
        _isler[is_id]["hata"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        _isler[is_id]["durum"] = "hata"
        _isler[is_id]["hata"] = f"{type(exc).__name__}: {exc}"
    finally:
        _is_yeri_birak()


@app.post("/api/is/oznitelik")
def oznitelik_baslat(dosya: str, arka_plan: BackgroundTasks):
    """Fotoğraftan öznitelik çıkarımını başlatır ve iş kimliği döner.

    Uzun iş kalıbı kullanılıyor çünkü süre öngörülemez: model bellekteyse 4-5
    saniye, değilse önce 69-110 saniyelik yükleme var. Sabit bir süre bekleyip
    "herhalde bitmiştir" demek burada kesinlikle yanlış cevap üretirdi.
    """
    yol = config.FOTO_DIZINI / Path(dosya).name
    if not yol.exists():
        raise HTTPException(404, "Fotoğraf bulunamadı")

    _is_yeri_ayir()
    is_id = uuid.uuid4().hex[:12]
    _isler[is_id] = {
        "id": is_id,
        "dosya": yol.name,
        "durum": "sirada",
        "model_hazir": _vlm.acik,
        "baslangic": datetime.now(timezone.utc).isoformat(),
    }
    arka_plan.add_task(_oznitelik_isi, is_id, yol)
    return {"is_id": is_id, "durum": "sirada", "model_hazir": _vlm.acik}


def _coklu_oznitelik_isi(is_id: str, foto_yolu: Path) -> None:
    """Fotoğraftaki bütün ürünleri bulur, kırpar ve her birine kimlik verir.

    Kırpma şart: kimlik fotoğrafın içerik özetinden üretiliyor. Kırpmadan üç
    ürün de aynı kimliği alırdı ve indeks A'da üçü de aynı vektörle temsil
    edilirdi — görsel arama onları birbirinden ayıramazdı. Her ürün kendi
    kırpığından kimlik alınca hem çakışma kalkıyor hem de her ürün kendi
    görseliyle indeksleniyor.
    """
    from src.ingest import vlm

    try:
        # --- 1. aşama: ürünleri ve kutularını bul ---
        _isler[is_id]["durum"] = (
            "model_yukleniyor" if not _vlm.acik else "cikariliyor"
        )
        bulunanlar = _vlm.oznitelik_cikar_coklu(foto_yolu)
        _isler[is_id]["bulunan_adet"] = len(bulunanlar)

        # --- 2. aşama: her kırpığa ayrı ayrı yeniden sor ---
        # Ölçüm (20 Ağustos, 4 ürünlü kompozit): tüm fotoğraftan çıkarımda
        # markaların 0/4'ü okundu, kırpıklardan 1/4 okundu — ve o bir tane
        # gözle okunabilen tek markaydı ("Being human"). Kırpınca ürüne düşen
        # piksel artıyor, yazı okunur hâle geliyor. Marka projenin en zayıf
        # metriği (%43) olduğu için bu ikinci geçiş bedelini hak ediyor:
        # ürün başına ~4.4 sn.
        baglanti = _baglanti()
        urunler = []
        for sira, ham in enumerate(bulunanlar, 1):
            _isler[is_id]["durum"] = f"okunuyor ({sira}/{len(bulunanlar)})"

            kirpik = vlm.kutuyu_kirp(foto_yolu, ham["kutu"])
            tampon = io.BytesIO()
            kirpik.save(tampon, format="JPEG", quality=92)
            veri = tampon.getvalue()

            kimlik = hashlib.sha256(veri).hexdigest()[:16]
            hedef = config.FOTO_DIZINI / f"{kimlik}.jpg"
            if not hedef.exists():
                hedef.write_bytes(veri)

            # Kırpıktan gelen okuma esas alınıyor; başarısız olursa 1. aşamanın
            # sonucu kullanılıyor. İkisi de olmazsa alan boş kalıp operatöre
            # bırakılıyor — uydurmak yerine.
            try:
                ince = _vlm.oznitelik_cikar(hedef) or {}
            except VlmHatasi:
                ince = {}

            def sec(alan: str):
                return ince.get(alan) or ham.get(alan)

            mevcut = stok.urun_getir(baglanti, kimlik)
            urunler.append({
                "sira": sira,
                "kimlik": kimlik,
                "dosya": hedef.name,
                "kutu": ham["kutu"],
                "kategori": sec("kategori"),
                "marka": sec("marka"),
                "renk": sec("renk"),
                "ayirt_edici": sec("ayirt_edici"),
                "zaten_kayitli": mevcut is not None,
            })

        _isler[is_id]["urunler"] = urunler
        _isler[is_id]["durum"] = "bitti"
    except VlmHatasi as exc:
        _isler[is_id]["durum"] = "hata"
        _isler[is_id]["hata"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        _isler[is_id]["durum"] = "hata"
        _isler[is_id]["hata"] = f"{type(exc).__name__}: {exc}"
    finally:
        _is_yeri_birak()


@app.post("/api/is/coklu-oznitelik")
def coklu_oznitelik_baslat(dosya: str, arka_plan: BackgroundTasks):
    """Tek fotoğraftaki birden çok ürünü ayrıştırmayı başlatır.

    Depoda ürünler çoğu zaman tek tek fotoğraflanmıyor: bir rafın ya da açılmış
    bir kolinin fotoğrafı çekiliyor. Tek ürün kipi böyle bir fotoğrafta yalnızca
    en büyük nesneyi anlatıp gerisini sessizce atıyordu.
    """
    yol = config.FOTO_DIZINI / Path(dosya).name
    if not yol.exists():
        raise HTTPException(404, "Fotoğraf bulunamadı")

    _is_yeri_ayir()
    is_id = uuid.uuid4().hex[:12]
    _isler[is_id] = {
        "id": is_id,
        "dosya": yol.name,
        "durum": "sirada",
        "model_hazir": _vlm.acik,
        "baslangic": datetime.now(timezone.utc).isoformat(),
    }
    arka_plan.add_task(_coklu_oznitelik_isi, is_id, yol)
    return {"is_id": is_id, "durum": "sirada", "model_hazir": _vlm.acik}


@app.get("/api/vlm/durum")
def vlm_durum():
    """İşçi süreç açık mı, ne kadar süredir boşta. Arayüz bekleme süresini
    buna göre dürüstçe yazıyor."""
    return _vlm.durum()


@app.post("/api/vlm/kapat")
def vlm_kapat():
    """İşçiyi elle kapatır — VRAM'e başka bir iş için ihtiyaç olduğunda."""
    _vlm.kapat()
    return {"acik": False}


class YeniUrun(BaseModel):
    # `dosya` yalnızca ad olarak kullanılıyor (Path(...).name), ama uzunluğu
    # yine sınırlı: sınırsız string gereksiz.
    dosya: str = Field(min_length=1, max_length=120)
    kategori: str = Field(min_length=1, max_length=200)
    marka: str | None = _METIN
    renk: str | None = _METIN
    urun_kodu: str | None = _METIN
    raf: str | None = _METIN
    adet: int = Field(1, ge=0, le=1_000_000)
    ayirt_edici: str | None = _METIN


@app.post("/api/urun", status_code=201)
def urun_olustur(govde: YeniUrun):
    """Yüklenmiş bir fotoğraftan stok kaydı oluşturur.

    Üç indeksin üçüne birden yazıyor. Yalnızca SQLite'a yazmak kaydı arama
    sonuçlarında görünmez yapardı — ürün "eklendi" görünüp aranınca çıkmazdı ki
    bu, hiç eklenmemiş olmasından daha kötü.
    """
    yol = config.FOTO_DIZINI / Path(govde.dosya).name
    if not yol.exists():
        raise HTTPException(404, "Fotoğraf bulunamadı; önce yükleyin")

    kimlik = chroma.urun_kimligi(yol)
    baglanti = _baglanti()
    if stok.urun_getir(baglanti, kimlik) is not None:
        raise HTTPException(409, "Bu fotoğraf zaten kayıtlı")

    oznitelik = {
        "kategori": govde.kategori,
        "marka": govde.marka,
        "renk": govde.renk,
        "ayirt_edici": govde.ayirt_edici,
    }
    metin = aciklama_uret(oznitelik, govde.renk)

    kayit = {
        "kimlik": kimlik,
        "dosya": yol.name,
        "kategori": govde.kategori,
        "marka": govde.marka,
        # Elle eklenen üründe marka da elle girilmiş oluyor (K11).
        "marka_kaynagi": "elle" if govde.marka else "bilinmiyor",
        "renk": govde.renk,
        "urun_kodu": govde.urun_kodu,
        "raf": govde.raf,
        "aciklama": metin,
        "eklendi": datetime.now(timezone.utc).isoformat(),
    }
    sqlite.ekle(baglanti, [kayit])
    if govde.adet != 1:
        stok.hareket_ekle(baglanti, kimlik, "duzeltme", govde.adet,
                          "ilk kayıt adedi", "operator")

    arayici = _arayici()
    meta = {
        "dosya": yol.name,
        "kategori": govde.kategori or "",
        "marka": govde.marka or "",
        "renk": govde.renk or "",
    }
    gorsel = arayici.gorsel_embedder.gorselleri_gom([yol], ilerleme=False)
    chroma.gorsel_koleksiyon().upsert(
        ids=[kimlik], embeddings=[gorsel[0].tolist()], metadatas=[meta],
    )
    if arayici.metin_embedder is not None:
        vek = arayici.metin_embedder.belgeleri_gom([metin])
        chroma.metin_koleksiyon().upsert(
            ids=[kimlik], embeddings=[vek[0].tolist()],
            metadatas=[{**meta, "aciklama": metin}],
        )

    return stok.urun_getir(baglanti, kimlik)


# --------------------------------------------------------------------------
# Arayüz dosyaları — en sona, API yollarını gölgelememesi için
# --------------------------------------------------------------------------

class TazeStatik(StaticFiles):
    """Arayüz dosyalarını her istekte doğrulatan statik sunucu.

    Varsayılan `StaticFiles` yalnızca ETag/Last-Modified veriyor; tarayıcı yine
    de dosyayı sormadan önbellekten kullanabiliyor. Geliştirme sırasında bu
    şuna yol açtı: `uygulama.js` değişti, sayfa eski sürümü çalıştırmaya devam
    etti ve "kod doğru ama arayüz güncellenmiyor" gibi görünen, aslında var
    olmayan hatalar arandı.

    `no-cache` dosyayı önbelleğe almayı engellemiyor; her kullanımdan önce
    sunucuya sormaya zorluyor. Değişmemişse 304 dönüyor, yani yerel bir
    uygulamada bedeli yok ama sürüm kayması ortadan kalkıyor.
    """

    def file_response(self, *args, **kwargs):
        yanit = super().file_response(*args, **kwargs)
        yanit.headers["Cache-Control"] = "no-cache, must-revalidate"
        return yanit


_arayuz = config.PROJE_KOK / "static"
if _arayuz.exists():
    app.mount("/", TazeStatik(directory=str(_arayuz), html=True), name="arayuz")
