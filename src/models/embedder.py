"""İki embedding modeli: görsel (İndeks A) ve metin (İndeks B).

VRAM bütçesi (8 GB):
  Görsel embedder   ~4.3 GB
  Metin embedder    ~1.1 GB
  Öznitelik VLM     ~2.5 GB (4-bit)

Sorgu anında iki embedder birlikte durabilir (~5.4 GB) — bir sorgu her iki
indekse de gittiği için buna zaten mecburuz. VLM ise yalnızca kayıt sırasında
ve ayrı bir süreçte çalışır, çünkü üçü aynı anda sığmaz.

Bu yüzden her iki sınıfta da `bosalt()` var: kayıt hattı, VLM'i çağırmadan önce
görsel embedder'ı bellekten düşürebilsin.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

from src import config


def _bellek_temizle() -> None:
    """Modeli sildikten sonra VRAM'i geri almaya çalışır.

    Tek başına yeterli DEĞİL — Python belleği hemen bırakmaz. Gerçek garanti
    süreç izolasyonundadır; bu sadece aynı süreç içinde elden geleni yapar.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class GorselEmbedder:
    """Fotoğraf ve metni aynı uzaya gömer — İndeks A.

    Kayıt sırasında fotoğraflar `gorselleri_gom` ile, arama sırasında sorgu
    metni `sorguyu_gom` ile gömülür. İkisi aynı uzaya düştüğü için metinle
    fotoğraf aranabiliyor.
    """

    def __init__(self, model_id: str = config.GORSEL_MODEL, boyut: int = config.GORSEL_BOYUT) -> None:
        self.model_id = model_id
        self.boyut = boyut
        self._model = None

    def _yukle(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[i] Görsel embedder yükleniyor: {self.model_id}")
            self._model = SentenceTransformer(self.model_id, truncate_dim=self.boyut)
            print(f"[+] Hazır. Cihaz: {self._model.device}, boyut: {self.boyut}")
        return self._model

    def gorselleri_gom(self, yollar: list[Path], ilerleme: bool = True) -> np.ndarray:
        """Fotoğrafları vektöre çevirir. Model dosya yolunu string olarak bekler."""
        model = self._yukle()
        return model.encode_document(
            [str(p.resolve()) for p in yollar],
            show_progress_bar=ilerleme,
            normalize_embeddings=True,
        )

    def sorguyu_gom(self, sorgular: list[str]) -> np.ndarray:
        model = self._yukle()
        return model.encode_query(sorgular, normalize_embeddings=True)

    def bosalt(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            _bellek_temizle()
            print("[i] Görsel embedder bellekten düşürüldü.")


class MetinEmbedder:
    """VLM açıklamalarını ve sorguları gömer — İndeks B.

    Görsel modellerin sıfat-isim bağını zayıf tutması yüzünden var. "Mavi valiz"
    sorgusu asıl burada doğru sonucu veriyor.

    e5 ailesi girdiye önek bekler ve bu önekler atlanırsa kalite belirgin düşer:
    saklanan metinler `passage: `, sorgular `query: ` ile başlar.
    """

    def __init__(self, model_id: str = config.METIN_MODEL) -> None:
        self.model_id = model_id
        self._model = None

    def _yukle(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"[i] Metin embedder yükleniyor: {self.model_id}")
            self._model = SentenceTransformer(self.model_id)
            print(f"[+] Hazır. Cihaz: {self._model.device}")
        return self._model

    def belgeleri_gom(self, metinler: list[str], ilerleme: bool = False) -> np.ndarray:
        model = self._yukle()
        return model.encode(
            [config.E5_BELGE_ONEKI + m for m in metinler],
            show_progress_bar=ilerleme,
            normalize_embeddings=True,
        )

    def sorguyu_gom(self, sorgular: list[str]) -> np.ndarray:
        model = self._yukle()
        return model.encode(
            [config.E5_SORGU_ONEKI + s for s in sorgular],
            normalize_embeddings=True,
        )

    def bosalt(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            _bellek_temizle()
            print("[i] Metin embedder bellekten düşürüldü.")
