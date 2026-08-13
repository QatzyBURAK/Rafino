"""Arama katmanı — tek indeks veya hibrit.

Aynı sorgu birden çok indekse gidip sonuçlar RRF ile birleştiriliyor. Her indeks
ayrı ayrı da çağrılabiliyor, çünkü ölçümde kazancın nereden geldiğini görmek
gerekiyor.

Kimlik olarak her yerde ChromaDB kaydının kimliği (fotoğrafın içerik özeti)
kullanılıyor; iki koleksiyon aynı kimliği paylaştığı için birleştirme sırasında
eşleştirme yapmaya gerek kalmıyor.
"""

from __future__ import annotations

from src import config
from src.db import chroma, sqlite
from src.models.embedder import GorselEmbedder, MetinEmbedder
from src.search.fusion import rrf_aciklamali


class Arayici:
    """Embedder'ları ve veritabanı bağlantısını açık tutar.

    Her sorguda model yüklemek çok pahalı (görsel embedder ~15 sn); bu sınıf
    bir kez kurulup tekrar tekrar kullanılıyor.
    """

    def __init__(self, gorsel: bool = True, metin: bool = True, fts: bool = True) -> None:
        self.gorsel_embedder = GorselEmbedder() if gorsel else None
        self.metin_embedder = MetinEmbedder() if metin else None
        self._sqlite = None
        if fts and config.SQLITE_YOLU.exists():
            self._sqlite = sqlite.baglan()

    # --- Tek indeksler ---

    def gorsel_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks A: sorgu metni -> fotoğraf vektörleri."""
        vektor = self.gorsel_embedder.sorguyu_gom([sorgu])[0]
        cevap = chroma.gorsel_koleksiyon().query(
            query_embeddings=[vektor.tolist()], n_results=k
        )
        return cevap["ids"][0]

    def metin_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks B: sorgu metni -> VLM açıklamalarının vektörleri."""
        koleksiyon = chroma.metin_koleksiyon()
        if koleksiyon.count() == 0:
            return []
        vektor = self.metin_embedder.sorguyu_gom([sorgu])[0]
        cevap = koleksiyon.query(query_embeddings=[vektor.tolist()], n_results=k)
        return cevap["ids"][0]

    def fts_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks C: anahtar kelime (BM25). Marka ve ürün kodu için."""
        if self._sqlite is None:
            return []
        return sqlite.ara(self._sqlite, sorgu, limit=k)

    # --- Hibrit ---

    def ara(
        self,
        sorgu: str,
        k: int = config.VARSAYILAN_SONUC,
        kullan: tuple[str, ...] = ("gorsel", "metin", "fts"),
        agirliklar: dict[str, float] | None = None,
        derinlik: int | None = None,
    ) -> list[dict]:
        """Seçilen indekslerde arar ve RRF ile birleştirir.

        derinlik: birleştirmeden önce her indeksten kaç sonuç alınacağı.
        k'dan büyük tutuluyor, çünkü bir indekste 8. sırada olan kayıt diğerinde
        1. sıradaysa birleşimde üste çıkabilir — dar alırsak o kaydı hiç görmeyiz.
        """
        derinlik = derinlik or max(k * 4, 20)
        siralamalar: dict[str, list[str]] = {}
        if "gorsel" in kullan and self.gorsel_embedder is not None:
            siralamalar["gorsel"] = self.gorsel_sirala(sorgu, derinlik)
        if "metin" in kullan and self.metin_embedder is not None:
            bulunan = self.metin_sirala(sorgu, derinlik)
            if bulunan:
                siralamalar["metin"] = bulunan
        if "fts" in kullan:
            bulunan = self.fts_sirala(sorgu, derinlik)
            # Boş sonucu RRF'e vermiyoruz: anahtar kelime araması eşleşme
            # bulamadığında sessiz kalmalı, sıralamayı seyreltmemeli.
            if bulunan:
                siralamalar["fts"] = bulunan

        if not siralamalar:
            return []
        return rrf_aciklamali(siralamalar, agirliklar=agirliklar)[:k]

    def kimlik_bilgisi(self, kimlikler: list[str]) -> dict[str, dict]:
        """Kimlikleri metadata'ya çevirir — sonuç kartı için."""
        if not kimlikler:
            return {}
        kayit = chroma.gorsel_koleksiyon().get(ids=kimlikler)
        return dict(zip(kayit["ids"], kayit["metadatas"]))

    def bosalt(self) -> None:
        if self.gorsel_embedder:
            self.gorsel_embedder.bosalt()
        if self.metin_embedder:
            self.metin_embedder.bosalt()
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None
