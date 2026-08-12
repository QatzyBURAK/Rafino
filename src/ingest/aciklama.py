"""Öznitelik JSON'undan Türkçe açıklama cümlesi üretir — İndeks B'nin girdisi.

İndeks B'nin varlık sebebi: görsel-metin modelleri sıfatı nesneye bağlamakta
zayıf ("mavi valiz" sorgusunda mavi etiketli siyah valizi de getiriyorlar). Dil
modelleri bu bağı çok daha iyi tutuyor. Elimizde VLM'in ürettiği yapısal
öznitelikler zaten var; bunları bir cümleye çevirip metin uzayında aramak,
sıfat-isim bağını doğal olarak koruyor.

Cümle şablonla üretiliyor, yeniden üretken bir model çağrılmıyor: çıktı
deterministik olsun ve aynı JSON hep aynı cümleyi versin diye.
"""

from __future__ import annotations

BOS_DEGERLER = {"", "bilinmiyor", "bilinmeyen", "yok", "none", "null", None}

# Marka bilgisinin nereden geldiği. Stok kaydında tutuluyor çünkü VLM'in okuduğu
# marka ile operatörün girdiği markanın güvenilirliği aynı değil ve arama
# sonuçlarında bu ayrım gösterilebilmeli.
MARKA_KAYNAKLARI = ("vlm", "elle", "barkod", "bilinmiyor")


def marka_kaynagi(oznitelik: dict) -> str:
    """VLM markayı okuyabildiyse 'vlm', okuyamadıysa 'bilinmiyor'.

    'bilinmiyor' kalan kayıtlar arayüzde operatöre gösterilecek ve elle
    tamamlandığında kaynak 'elle' olacak. Ölçüm, VLM'in markayı ürünlerin ancak
    %43'ünde okuyabildiğini gösterdi; kalan %57 için elle giriş bir eksiklik
    değil, tasarımın parçası.
    """
    return "bilinmiyor" if _gecerli(oznitelik.get("marka")) is None else "vlm"


def _gecerli(deger) -> str | None:
    """Değeri temizler; anlamsızsa None döner."""
    if deger is None:
        return None
    metin = str(deger).strip()
    return None if metin.lower() in BOS_DEGERLER else metin


def aciklama_uret(oznitelik: dict, renk: str | None = None) -> str:
    """Öznitelik sözlüğünden aranabilir Türkçe cümle kurar.

    `renk` ayrı parametre, çünkü hangi kaynaktan geldiği (piksel veya VLM)
    değişebiliyor; bu modül kaynağı bilmek zorunda değil.

    Örnek çıktı:
        "mavi deri el çantası, Murcia marka. ön cebi fermuarlı. sağlam."
    """
    kategori = _gecerli(oznitelik.get("kategori")) or "ürün"
    renk = _gecerli(renk) or _gecerli(oznitelik.get("renk"))
    malzeme = _gecerli(oznitelik.get("malzeme"))
    marka = _gecerli(oznitelik.get("marka"))
    ayirt = _gecerli(oznitelik.get("ayirt_edici"))
    durum = _gecerli(oznitelik.get("durum"))

    # Ana öbek: sıfatlar isimden önce, Türkçe söz dizimine uygun.
    # Sıralama önemli — "mavi deri el çantası", "deri mavi el çantası" değil.
    parcalar = [p for p in (renk, malzeme, kategori) if p]
    cumle = " ".join(parcalar)

    if marka:
        cumle += f", {marka} marka"
    cumle += "."

    if ayirt:
        cumle += f" {ayirt}."
    if durum and durum != "saglam":
        # Sadece hasarlı/bilinmeyen durumu yaz; "sağlam" varsayılan olduğu için
        # her cümleye eklemek metni seyreltir ve aramayı bozar.
        cumle += f" {durum}."

    return cumle


def aranabilir_metin(oznitelik: dict, renk: str | None = None) -> str:
    """İndeks B'ye gidecek nihai metin.

    Şimdilik açıklamanın kendisi. Ayrı fonksiyon olarak duruyor çünkü ileride
    marka/kategori tekrarı gibi ağırlıklandırmalar burada denenecek ve
    `aciklama_uret` insan tarafından okunabilir kalmalı.
    """
    return aciklama_uret(oznitelik, renk)
