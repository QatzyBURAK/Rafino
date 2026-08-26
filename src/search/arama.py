"""Arama katmanı — tek indeks veya hibrit.

Aynı sorgu birden çok indekse gidip sonuçlar RRF ile birleştiriliyor. Her indeks
ayrı ayrı da çağrılabiliyor, çünkü ölçümde kazancın nereden geldiğini görmek
gerekiyor.

Kimlik olarak her yerde ChromaDB kaydının kimliği (fotoğrafın içerik özeti)
kullanılıyor; iki koleksiyon aynı kimliği paylaştığı için birleştirme sırasında
eşleştirme yapmaya gerek kalmıyor.
"""

from __future__ import annotations

import threading

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
        # FTS bağlantısı iş parçacığı başına açılıyor; bkz. `_fts_baglantisi`.
        self._fts_acik = bool(fts and config.SQLITE_YOLU.exists())
        self._yerel = threading.local()

    # --- Tek indeksler ---

    @staticmethod
    def _esikle(cevap: dict, esik: float) -> list[str]:
        """Eşikten uzak komşuları atar.

        Vektör araması "bulamadım" diyemez; her zaman en yakın k kaydı döner.
        Depoda olmayan bir ürün arandığında bu, alakasız sonuçlardan oluşan
        ikna edici bir liste üretiyor. Mesafeye bakıp uzak olanları atmak,
        sisteme "bilmiyorum" deme imkânı veriyor.
        """
        kimlikler = cevap["ids"][0]
        mesafeler = cevap.get("distances", [[]])[0]
        if not mesafeler:
            return kimlikler
        return [k for k, m in zip(kimlikler, mesafeler) if m <= esik]

    def gorsel_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks A: sorgu metni -> fotoğraf vektörleri."""
        vektor = self.gorsel_embedder.sorguyu_gom([sorgu])[0]
        cevap = chroma.gorsel_koleksiyon().query(
            query_embeddings=[vektor.tolist()], n_results=k
        )
        return self._esikle(cevap, config.GORSEL_ALAKA_ESIGI)

    def metin_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks B: sorgu metni -> VLM açıklamalarının vektörleri."""
        koleksiyon = chroma.metin_koleksiyon()
        if koleksiyon.count() == 0:
            return []
        vektor = self.metin_embedder.sorguyu_gom([sorgu])[0]
        cevap = koleksiyon.query(query_embeddings=[vektor.tolist()], n_results=k)
        return self._esikle(cevap, config.METIN_ALAKA_ESIGI)

    def _fts_baglantisi(self):
        """Bu iş parçacığına ait FTS bağlantısını döner, yoksa açar.

        `Arayici` yaşam döngüsünde bir kez kuruluyor, ama `ara()` her istekte
        FastAPI'nin iş parçacığı havuzundan çağrılıyor. Tek paylaşılan bağlantı
        "SQLite objects created in a thread can only be used in that same
        thread" hatası veriyordu: indeks C sessizce boş dönmüyor, arama ucunu
        tamamen 500'e düşürüyordu. Aynı kalıp api/main.py `_baglanti()`'de var.
        """
        if not self._fts_acik:
            return None
        baglanti = getattr(self._yerel, "baglanti", None)
        if baglanti is None:
            baglanti = self._yerel.baglanti = sqlite.baglan()
        return baglanti

    def fts_sirala(self, sorgu: str, k: int) -> list[str]:
        """İndeks C: anahtar kelime (BM25). Marka ve ürün kodu için."""
        baglanti = self._fts_baglantisi()
        if baglanti is None:
            return []
        return sqlite.ara(baglanti, sorgu, limit=k)

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
        # Ağırlık verilmediyse ölçülmüş varsayılan kullanılıyor. Eşit ağırlık,
        # indeks C'nin kesin marka eşleşmesini görsel gürültüyle aynı kefeye
        # koyuyordu; bkz. config.RRF_AGIRLIKLARI.
        if agirliklar is None:
            agirliklar = config.RRF_AGIRLIKLARI

        derinlik = derinlik or max(k * 4, 20)
        siralamalar: dict[str, list[str]] = {}
        if "gorsel" in kullan and self.gorsel_embedder is not None:
            # Boş sonuç RRF'e verilmiyor; eşiği geçen kayıt yoksa bu indeks
            # sessiz kalmalı (aynı kural aşağıda metin ve fts için de var).
            bulunan = self.gorsel_sirala(sorgu, derinlik)
            if bulunan:
                siralamalar["gorsel"] = bulunan
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
        # Yalnızca bu parçacığın bağlantısı kapatılıyor; diğerlerini süreç
        # sonlanırken işletim sistemi topluyor.
        self._fts_acik = False
        baglanti = getattr(self._yerel, "baglanti", None)
        if baglanti is not None:
            baglanti.close()
            self._yerel.baglanti = None
