"""Reciprocal Rank Fusion — birden çok indeksin sonucunu tek sıralamada birleştirir.

Neden skor değil SIRA toplanıyor:
Farklı modellerin benzerlik değerleri kıyaslanamaz. Görsel indekste 0.71
mükemmel bir eşleşmeyken metin indeksinde vasat olabilir; ölçekleri ve
dağılımları farklıdır. Skorları normalize etmeye çalışmak da kırılgan, çünkü
dağılım sorgudan sorguya değişiyor. "1. sıradaydı" ifadesi ise her indekste
aynı şeyi anlatır.

    skor(kayit) = toplam  1 / (K + sira)

K sabiti (varsayılan 60), tek bir indeksin ilk sıralarının sonucu tek başına ele
geçirmesini engelliyor: 1. ile 2. sıra arasındaki fark 1/61 - 1/62 = 0.00026
gibi küçük kalıyor, dolayısıyla iki indekste birden üst sıralarda çıkan bir
kayıt, tek indekste 1. olan kaydı geçebiliyor. RRF'in asıl faydası bu — tek bir
indeksin yanılması sonucu bozmuyor.
"""

from __future__ import annotations

from src import config


def rrf(
    siralamalar: dict[str, list[str]],
    k: int = config.RRF_K,
    agirliklar: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Sıralanmış kimlik listelerini RRF ile birleştirir.

    siralamalar: {"gorsel": [id1, id2, ...], "metin": [...], "fts": [...]}
                 her liste en iyiden en kötüye sıralı olmalı.
    agirliklar:  indeks başına çarpan. Verilmezse hepsi eşit.

    Döner: (kimlik, skor) çiftleri, skora göre azalan.
    """
    agirliklar = agirliklar or {}
    skorlar: dict[str, float] = {}
    for ad, kimlikler in siralamalar.items():
        agirlik = agirliklar.get(ad, 1.0)
        for sira, kimlik in enumerate(kimlikler, start=1):
            skorlar[kimlik] = skorlar.get(kimlik, 0.0) + agirlik / (k + sira)
    return sorted(skorlar.items(), key=lambda x: -x[1])


def rrf_aciklamali(
    siralamalar: dict[str, list[str]],
    k: int = config.RRF_K,
    agirliklar: dict[str, float] | None = None,
) -> list[dict]:
    """rrf() ile aynı, ama her kaydın hangi indeksten kaçıncı sırada geldiğini de döner.

    Hata ayıklama ve rapor için: "bu ürün neden 1. çıktı?" sorusunun cevabı.
    """
    agirliklar = agirliklar or {}
    detay: dict[str, dict] = {}
    for ad, kimlikler in siralamalar.items():
        agirlik = agirliklar.get(ad, 1.0)
        for sira, kimlik in enumerate(kimlikler, start=1):
            kayit = detay.setdefault(kimlik, {"kimlik": kimlik, "skor": 0.0, "siralar": {}})
            kayit["skor"] += agirlik / (k + sira)
            kayit["siralar"][ad] = sira
    return sorted(detay.values(), key=lambda x: -x["skor"])
