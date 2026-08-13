"""Kopya ve benzer ürün tespiti.

Depolarda gerçek bir problem: aynı ürün farklı zamanlarda, farklı kişiler
tarafından tekrar tekrar kaydediliyor ve stok sayısı şişiyor. Elle girişte bunu
yakalamak imkânsıza yakın; görsel arama mimarisi ise bunu neredeyse bedavaya
veriyor, çünkü "bu fotoğrafa en çok benzeyen kayıtlar" sorgusu zaten var.

İki farklı durum ayrılıyor:

  BİREBİR KOPYA — aynı dosya. İçerik özeti aynı olduğu için kimlik çakışır ve
                  kayıt zaten tekilleşir; ayrıca tespit gerekmez.

  AYNI ÜRÜN     — farklı açıdan/ışıkta çekilmiş aynı fiziksel ürün. Kimlikler
                  farklı, vektörler çok yakın. Asıl yakalanması gereken bu.

  BENZER ÜRÜN   — aynı modelin başka rengi, ya da aynı kategoriden başka ürün.
                  Kaydedilmeli ama operatöre "buna benzer şunlar var" diye
                  gösterilmesi faydalı.

Eşikler ölçümle belirlendi (bkz. scripts/test_kopya.py), tahminle değil.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.db import chroma
from src.models.embedder import GorselEmbedder

# Eşikler ÖLÇÜLDÜ (scripts/test_kopya.py, 12 Ağustos 2026, 20 ürün):
#
#   Pozitif (aynı ürünün döndürülmüş/kırpılmış/ışığı değişmiş hâli)
#       n=100   min 0.896   ort 0.984
#   Zor negatif (AYNI kategoride farklı ürünler)
#       n=10    min 0.230   ort 0.513   max 0.873
#   Kolay negatif (farklı kategoriler)
#       n=180   ort 0.143
#
# Dağılımlar ayrık: pozitiflerin en düşüğü (0.896) > zor negatiflerin en
# yükseği (0.873). Eşik ikisinin ortasına konuldu.
#
# İlk tahmin 0.92'ydi ve ölçüm onu düzeltti: %50 küçültülmüş fotoğraf 0.896
# benzerlik veriyor, yani 0.92 eşiğiyle gerçek bir kopya kaçırılacaktı.
#
# UYARI: zor negatif örneklemi yalnızca 10 çift. Stok büyüdükçe bu ölçüm
# tekrarlanmalı; aynı modelin farklı renkleri eklendiğinde negatiflerin üst
# sınırı yükselebilir ve eşik yukarı çekilmesi gerekebilir.
AYNI_URUN_ESIGI = 0.88

# Bilgilendirme bandının alt sınırı. Zor negatiflerin ortalaması 0.513, en
# yükseği 0.873; 0.70 bu aralığın üst kısmını yakalıyor, yani "aynı kategoriden
# her ürün" için değil yalnızca gerçekten yakın olanlar için uyarı çıkıyor.
BENZER_URUN_ESIGI = 0.70


@dataclass
class KopyaBulgusu:
    """Bir kayıt adayı için kopya kontrolü sonucu."""

    durum: str  # "birebir" | "ayni_urun" | "benzer" | "yeni"
    eslesmeler: list[dict]  # [{kimlik, benzerlik, dosya, ...}]

    @property
    def onay_gerekli(self) -> bool:
        """Operatöre sorulmalı mı?

        "benzer" durumunda sormuyoruz: aynı kategoriden her ürün birbirine
        benzer ve her kayıtta soru sormak sistemi kullanılamaz hale getirir.
        """
        return self.durum in {"birebir", "ayni_urun"}


def kopya_kontrol(
    foto_yolu: Path,
    embedder: GorselEmbedder | None = None,
    aday_sayisi: int = 5,
) -> KopyaBulgusu:
    """Fotoğrafı kaydetmeden önce stokta benzeri var mı diye bakar."""
    koleksiyon = chroma.gorsel_koleksiyon()
    if koleksiyon.count() == 0:
        return KopyaBulgusu("yeni", [])

    kimlik = chroma.urun_kimligi(foto_yolu)
    mevcut = koleksiyon.get(ids=[kimlik])
    if mevcut["ids"]:
        # Aynı dosya daha önce kaydedilmiş; vektör hesaplamaya gerek yok.
        return KopyaBulgusu("birebir", [{
            "kimlik": kimlik,
            "benzerlik": 1.0,
            **(mevcut["metadatas"][0] or {}),
        }])

    embedder = embedder or GorselEmbedder()
    vektor = embedder.gorselleri_gom([foto_yolu], ilerleme=False)[0]
    cevap = koleksiyon.query(
        query_embeddings=[vektor.tolist()],
        n_results=min(aday_sayisi, koleksiyon.count()),
    )

    eslesmeler = [
        {"kimlik": kid, "benzerlik": round(1 - mesafe, 4), **(meta or {})}
        for kid, mesafe, meta in zip(
            cevap["ids"][0], cevap["distances"][0], cevap["metadatas"][0]
        )
    ]
    if not eslesmeler:
        return KopyaBulgusu("yeni", [])

    en_yakin = eslesmeler[0]["benzerlik"]
    if en_yakin >= AYNI_URUN_ESIGI:
        durum = "ayni_urun"
    elif en_yakin >= BENZER_URUN_ESIGI:
        durum = "benzer"
    else:
        durum = "yeni"

    ilgili = [e for e in eslesmeler if e["benzerlik"] >= BENZER_URUN_ESIGI]
    return KopyaBulgusu(durum, ilgili or eslesmeler[:1])


def mesaj(bulgu: KopyaBulgusu) -> str:
    """Operatöre gösterilecek metin.

    Sessiz davranmıyoruz: sistem bir kaydı kopya sandığında bunu açıkça söylüyor.
    Sessiz birleştirme, yanlış birleştirdiğinde fark edilmez ve stok bozulur.
    """
    if bulgu.durum == "yeni":
        return "Stokta benzeri yok, yeni kayıt."
    if bulgu.durum == "birebir":
        e = bulgu.eslesmeler[0]
        return (f"Bu fotoğraf zaten kayıtlı ({e.get('dosya', '?')}"
                f"{', raf ' + e['raf'] if e.get('raf') else ''}). "
                f"Adet artırılsın mı?")
    if bulgu.durum == "ayni_urun":
        e = bulgu.eslesmeler[0]
        return (f"Bu ürün stokta görünüyor: {e.get('dosya', '?')}"
                f"{', raf ' + e['raf'] if e.get('raf') else ''} "
                f"(benzerlik {e['benzerlik']:.2f}). "
                f"Aynı ürün mü, yoksa yeni kayıt mı?")
    return (f"Stokta {len(bulgu.eslesmeler)} benzer ürün var; "
            f"en yakını {bulgu.eslesmeler[0]['benzerlik']:.2f}. Kayıt sürüyor.")
