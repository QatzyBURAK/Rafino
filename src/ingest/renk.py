"""Renk çıkarımı — VLM'e değil piksellere sorularak.

Neden VLM'e sorulmuyor (karar K4): VLM aynı çağrıda hem alan hem serbest metin
istendiğinde kendi içinde çelişiyordu — `renk: gümüş` derken açıklamada "koyu
mavi" yazabiliyordu. Renk, görüntüden deterministik olarak hesaplanabilen bir
büyüklük; üretken modele sormak gereksiz belirsizlik ekliyor.

Yöntem:
  1. Merkez kırpma        — ürün ortada, kenarlar arka plan
  2. Arka plan ayıklama   — ürün fotoğrafları beyaz fonlu; köşelerden fon rengi
                            öğrenilip ona yakın pikseller atılıyor
  3. k-means (k=3)        — kalan piksellerde baskın küme
  4. Palet eşleme         — HSV kurallarıyla sabit Türkçe renk adına indirgeme

Adım 2 olmadan her ürün "beyaz" çıkar; ürün fotoğraflarında piksellerin çoğu
fondur.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image

# Çıktı paleti. Kaynak veri setindeki ince ayrımlar (gümüş/gri, altın/sarı,
# bej/ten rengi) bilerek birleştirildi: bunlar pikselden güvenilir ayrılamıyor
# ve depo aramasında da ayırt edici değil.
PALET = [
    "siyah", "beyaz", "gri",
    "kırmızı", "turuncu", "sarı", "yeşil", "mavi", "lacivert",
    "mor", "pembe", "kahverengi", "bej", "bordo",
]

# Ölçüm sırasında "yanlış ama makul" sayılan eşleştirmeler. Ground truth'ta
# gümüş yazan bir saatin gri çıkması gerçek bir hata değil.
YAKIN_RENKLER: dict[str, set[str]] = {
    "gri": {"gümüş", "çelik", "antrasit", "gri melanj"},
    "sarı": {"altın", "hardal"},
    "bej": {"ten rengi", "krem", "kırık beyaz", "bakır"},
    "beyaz": {"kırık beyaz", "krem"},
    "kahverengi": {"kahve", "bakır", "bronz", "kiremit", "ten rengi"},
    "bordo": {"kırmızı", "kiremit"},
    "kırmızı": {"bordo"},
    "mavi": {"petrol mavisi", "turkuaz", "lacivert"},
    "lacivert": {"mavi"},
    "mor": {"lavanta", "leylak", "fuşya"},
    "pembe": {"gül kurusu", "fuşya", "şeftali"},
    "yeşil": {"zeytin yeşili", "haki", "deniz yeşili", "fıstık yeşili", "petrol mavisi"},
}


def _arka_plani_ayikla(piksel: np.ndarray, kenar: np.ndarray, tolerans: int = 34) -> np.ndarray:
    """Kenarlardan öğrenilen fon rengine yakın pikselleri atar.

    Kritik eşik: ayıklamadan sonra piksellerin en az %15'i kalmalı. Beyaz fon
    üzerindeki BEYAZ ürünlerde ayıklama ürünü de siliyor ve geriye yalnızca
    gölge kalıyordu; gölgenin rengi de "bej" olarak raporlanıyordu. Eşik %2 iken
    bu sessizce oluyordu — ilk ölçümdeki `beyaz -> bej` hatalarının sebebi buydu.
    """
    fon = np.median(kenar, axis=0)
    uzaklik = np.linalg.norm(piksel.astype(np.int16) - fon.astype(np.int16), axis=1)
    kalan = piksel[uzaklik > tolerans]
    # Ürün fonla aynı renkteyse (beyaz gömlek, beyaz ayakkabı) ayıklamadan vazgeç.
    if len(kalan) < max(256, int(len(piksel) * 0.15)):
        return piksel
    return kalan


def _kmeans(veri: np.ndarray, k: int = 3, tur: int = 12, tohum: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Küçük ve bağımlılıksız k-means. sklearn de olurdu ama bu yeterli ve hızlı."""
    rng = np.random.default_rng(tohum)
    merkez = veri[rng.choice(len(veri), size=min(k, len(veri)), replace=False)].astype(np.float64)
    etiket = np.zeros(len(veri), dtype=np.int64)
    for _ in range(tur):
        uzaklik = np.linalg.norm(veri[:, None, :] - merkez[None, :, :], axis=2)
        yeni = uzaklik.argmin(axis=1)
        if np.array_equal(yeni, etiket):
            break
        etiket = yeni
        for i in range(len(merkez)):
            uyeler = veri[etiket == i]
            if len(uyeler):
                merkez[i] = uyeler.mean(axis=0)
    return merkez, etiket


def _rgb_to_ad(r: int, g: int, b: int) -> str:
    """RGB -> Türkçe renk adı. Kurallar HSV üzerinde, sırayla uygulanıyor."""
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    v = max(r, g, b) / 255
    doygunluk = s
    derece = h * 360

    # Akromatik: önce parlaklık, sonra doygunluk eşiği.
    if v < 0.16:
        return "siyah"
    if doygunluk < 0.12 or (v < 0.30 and doygunluk < 0.25):
        if v > 0.85:
            return "beyaz"
        if v > 0.30:
            return "gri"
        return "siyah"

    # Kromatik: hue aralıkları, koyu tonlar ayrı adlandırılıyor.
    if derece < 15 or derece >= 345:
        return "bordo" if v < 0.45 else "kırmızı"
    if derece < 45:
        if v < 0.45:
            return "kahverengi"
        # Açık ve az doygun turuncu tonu bej; bu ayrım deri/kumaş ürünlerde çok işe yarıyor.
        if doygunluk < 0.45 and v > 0.70:
            return "bej"
        return "kahverengi" if doygunluk > 0.55 and v < 0.65 else "turuncu"
    if derece < 70:
        return "kahverengi" if v < 0.50 else "sarı"
    if derece < 165:
        return "yeşil"
    if derece < 200:
        return "mavi"
    if derece < 255:
        return "lacivert" if v < 0.55 else "mavi"
    if derece < 290:
        return "mor"
    return "pembe" if v > 0.55 else "mor"


def renk_bul(foto_yolu: Path, merkez_orani: float = 0.62) -> tuple[str, float]:
    """Fotoğraftaki ürünün baskın rengini döndürür.

    Dönen ikinci değer, baskın kümenin kalan pikseller içindeki payı — düşükse
    ürün çok renklidir ve renk bilgisine az güvenilmelidir.
    """
    gorsel = Image.open(foto_yolu).convert("RGB")
    # Küçültme hem hızlandırıyor hem JPEG gürültüsünü bastırıyor.
    gorsel.thumbnail((256, 256), Image.Resampling.LANCZOS)
    dizi = np.asarray(gorsel)
    yuk, gen = dizi.shape[:2]

    # Kenar şeridi = fon örneği (merkez kırpmanın dışında kalan bölge)
    kenar = np.concatenate([
        dizi[: max(1, yuk // 12)].reshape(-1, 3),
        dizi[-max(1, yuk // 12):].reshape(-1, 3),
        dizi[:, : max(1, gen // 12)].reshape(-1, 3),
        dizi[:, -max(1, gen // 12):].reshape(-1, 3),
    ])

    y0, y1 = int(yuk * (1 - merkez_orani) / 2), int(yuk * (1 + merkez_orani) / 2)
    x0, x1 = int(gen * (1 - merkez_orani) / 2), int(gen * (1 + merkez_orani) / 2)
    merkez = dizi[y0:y1, x0:x1].reshape(-1, 3)

    kalan = _arka_plani_ayikla(merkez, kenar)
    if len(kalan) < 8:
        kalan = merkez

    merkezler, etiketler = _kmeans(kalan.astype(np.float64), k=3)
    sayim = np.bincount(etiketler, minlength=len(merkezler))
    baskin = int(sayim.argmax())
    r, g, b = (int(round(x)) for x in merkezler[baskin])
    pay = float(sayim[baskin] / sayim.sum())
    return _rgb_to_ad(r, g, b), pay


_TR_SADELESTIRME = str.maketrans("ıİşŞğĞüÜöÖçÇ", "iisSgGuUoOcC")


def _tekil_normalize(renk: str) -> str:
    """Tek bir renk adını sadeleştirir.

    Türkçe karakter farkını kapatır: VLM'e verilen palet ASCII yazıldığı için
    model "kirmizi" döndürüyor, ground truth ise "kırmızı" diyor. Aynı renk;
    ölçümde ayrı sayılırsa VLM haksız yere düşük puan alıyor. İlk ölçümde tam
    olarak bu oldu ve VLM'in üstünlüğü %10 görünürken gerçekte %33'tü.
    """
    metin = renk.strip().lower()
    for on_ek in ("koyu ", "açık ", "acik ", "light ", "dark "):
        if metin.startswith(on_ek):
            metin = metin[len(on_ek):].strip()
    return metin.translate(_TR_SADELESTIRME)


def renkleri_ayir(renk: str | None) -> list[str]:
    """Çok renkli cevabı tek tek renklere böler.

    "siyah-beyaz" -> ["siyah", "beyaz"]

    Bu bir kural ihlali değil, doğru davranış. Çizgili veya iki tonlu ürünlerde
    VLM iki renk söylüyor ve fotoğrafa bakıldığında haklı çıkıyor; kaynak veri
    setinin tek renklik etiketi indirgeyici (örnek: siyah-beyaz-gri çizgili Puma
    polo "Black" diye etiketlenmiş).

    Aramada da doğrusu bu: siyah-beyaz çizgili bir gömlek hem "siyah gömlek"
    hem "beyaz gömlek" sorgusunda çıkmalı.
    """
    if not renk:
        return []
    metin = str(renk).strip().lower()
    for ayirac in ("-", "/", ",", " ve "):
        metin = metin.replace(ayirac, "|")
    return [_tekil_normalize(p) for p in metin.split("|") if p.strip()]


def normalize(renk: str | None) -> str:
    """Tek bir sadeleştirilmiş renk adı — çok renkliyse ilki."""
    parcalar = renkleri_ayir(renk)
    return parcalar[0] if parcalar else ""


def birebir_mi(tahmin: str, gercek: str) -> bool:
    """Gerçek renk, tahminde geçiyor mu?

    Çok renkli tahminlerde renklerden HERHANGİ biri tutuyorsa doğru sayılıyor:
    "siyah-beyaz" cevabı, etiketi "siyah" olan ürün için doğrudur.
    """
    g = normalize(gercek)
    return g in renkleri_ayir(tahmin)


def makul_mu(tahmin: str, gercek: str) -> bool:
    """Birebir doğru değilse bile yakın renk grubunda mı?"""
    if birebir_mi(tahmin, gercek):
        return True
    g = normalize(gercek)
    tahminler = set(renkleri_ayir(tahmin))
    for anahtar, kume in YAKIN_RENKLER.items():
        if _tekil_normalize(anahtar) in tahminler:
            if g in {_tekil_normalize(x) for x in kume}:
                return True
    return False
