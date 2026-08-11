"""ChromaDB bağlantısı ve koleksiyonlar.

İki ayrı koleksiyon tutulur:
  urun_gorsel — fotoğraf vektörleri (İndeks A)
  urun_metin  — VLM açıklamasının vektörleri (İndeks B)

Aynı ürün her iki koleksiyonda da AYNI kimlikle durur, böylece sonuçlar
birleştirilirken eşleştirme yapmaya gerek kalmaz.
"""

import hashlib
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from src import config

_istemci: chromadb.ClientAPI | None = None


def istemci() -> chromadb.ClientAPI:
    """Kalıcı ChromaDB istemcisi. Süreç başına tek örnek."""
    global _istemci
    if _istemci is None:
        config.dizinleri_hazirla()
        _istemci = chromadb.PersistentClient(path=str(config.CHROMA_DIZINI))
    return _istemci


def gorsel_koleksiyon() -> Collection:
    return istemci().get_or_create_collection(
        config.GORSEL_KOLEKSIYON,
        metadata={"hnsw:space": config.MESAFE_METRIGI},
    )


def metin_koleksiyon() -> Collection:
    return istemci().get_or_create_collection(
        config.METIN_KOLEKSIYON,
        metadata={"hnsw:space": config.MESAFE_METRIGI},
    )


def urun_kimligi(foto_yolu: Path) -> str:
    """Fotoğrafın içeriğinden türetilen sabit kimlik.

    Dosya yolu kimlik olarak KULLANILMAZ: klasör taşınınca veya dosya adı
    değişince tüm veritabanı kırılır — eski demoda tam olarak bu oldu.

    İçerik özeti kullanmanın ikinci faydası: byte düzeyinde birebir aynı iki
    fotoğraf aynı kimliği alır, yani aynı dosyanın kopyaları kendiliğinden
    tekilleşir. (Aynı ürünün FARKLI açıdan çekilmiş fotoğrafı ayrı bir sorun;
    o vektör benzerliğiyle yakalanacak.)
    """
    ozet = hashlib.sha256(foto_yolu.read_bytes()).hexdigest()
    return ozet[:16]


def sifirla() -> None:
    """Her iki koleksiyonu da siler. Yeniden indeksleme öncesi kullanılır."""
    c = istemci()
    for ad in (config.GORSEL_KOLEKSIYON, config.METIN_KOLEKSIYON):
        try:
            c.delete_collection(ad)
        except Exception:  # noqa: BLE001 - koleksiyon yoksa sorun değil
            pass
