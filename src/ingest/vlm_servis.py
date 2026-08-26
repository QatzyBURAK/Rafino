"""Sıcak VLM işçisini yöneten katman — API bunu kullanıyor.

İşçi süreç (`scripts/vlm_isci.py`) modeli bir kez yükleyip açık tutuyor. Bu modül
onu ilk ihtiyaçta doğuruyor, isteklerini sıraya sokuyor ve bir süre kullanılmazsa
kapatıp VRAM'i geri veriyor.

Neden boşta kapatma var: model 4-bit hâlde bile ~3.9 GB tutuyor ve bu makinede
GPU 8 GB. Arama embedder'ları zaten yüklü; VLM sonsuza kadar açık kalırsa başka
hiçbir şey sığmıyor. Depoda ürün ekleme seyrek ve öbekli bir iş — on ürün arka
arkaya ekleniyor, sonra saatlerce dokunulmuyor. Boşta kapatma tam bu şekle uyuyor:
öbeğin ilk fotoğrafı yükleme bedelini ödüyor, gerisi hızlı, öbek bitince VRAM
serbest kalıyor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from src import config

# Boşta bekleme süresi. Kısa tutulursa öbek ortasında model boşuna kapanıp
# yeniden yükleniyor; uzun tutulursa VRAM gereksiz yere tutuluyor.
BOSTA_SINIR_SN = 10 * 60

# Model yüklemesi ölçülen en kötü hâlde 110 sn. 300 sn, yavaş diskten ilk
# okumaya da yer bırakıyor ve donmuş bir sürece sonsuza kadar beklemiyor.
YUKLEME_ZAMAN_ASIMI_SN = 300

# Tek fotoğrafın işlenmesi 4-5 sn. 120 sn fazlasıyla geniş; aşılırsa süreç
# gerçekten takılmıştır.
ISTEK_ZAMAN_ASIMI_SN = 120

# Çoklu kipte model her ürün için ayrı nesne üretiyor: dört ürünlü fotoğrafta
# 22 sn ölçüldü. Kalabalık bir raf fotoğrafı bunun katı olabileceği için sınır
# geniş tutuluyor.
ISTEK_ZAMAN_ASIMI_SN_COKLU = 300


class VlmHatasi(RuntimeError):
    """İşçi süreç bir isteği karşılayamadı."""


class VlmServisi:
    """İşçi sürecin ömrünü ve erişimini yönetir.

    Tek bir kilit tüm süreci koruyor: işçi aynı anda tek istek işleyebiliyor
    (tek GPU, tek model) ve protokol satır sıralı. Paralel istek göndermek
    yanıtları birbirine karıştırırdı.
    """

    def __init__(self, bosta_sinir_sn: int = BOSTA_SINIR_SN) -> None:
        self._kilit = threading.Lock()
        self._surec: subprocess.Popen | None = None
        self._son_kullanim = 0.0
        self._bosta_sinir = bosta_sinir_sn
        self._yukleme_sn: float | None = None
        self._zamanlayici: threading.Timer | None = None

    # --- durum ---

    @property
    def acik(self) -> bool:
        return self._surec is not None and self._surec.poll() is None

    def durum(self) -> dict:
        return {
            "acik": self.acik,
            "yukleme_sn": self._yukleme_sn,
            "bosta_sn": (
                round(time.monotonic() - self._son_kullanim)
                if self.acik else None
            ),
        }

    # --- yaşam döngüsü ---

    def _baslat(self) -> None:
        """İşçiyi doğurur ve 'hazır' satırını bekler. Kilit ÇAĞIRANDA olmalı."""
        betik = config.PROJE_KOK / "scripts" / "vlm_isci.py"
        # PYTHONIOENCODING olmadan çocuk süreç Windows'ta stdout'u cp1254 ile
        # kodluyor, burada ise UTF-8 okunuyordu; Türkçe içeren her yanıt
        # "'utf-8' codec can't decode byte 0xf6" ile düşüyordu.
        ortam = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self._surec = subprocess.Popen(
            [sys.executable, str(betik)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # işçinin günlükleri sunucu günlüğüne aksın
            text=True,
            encoding="utf-8",
            errors="replace",
            env=ortam,
            cwd=str(config.PROJE_KOK),
        )
        yanit = self._satir_oku(YUKLEME_ZAMAN_ASIMI_SN)
        if not yanit.get("hazir"):
            hata = yanit.get("hata", "bilinmeyen sebep")
            self._oldur()
            raise VlmHatasi(f"VLM yüklenemedi: {hata}")
        self._yukleme_sn = yanit.get("yukleme_sn")

    def _satir_oku(self, zaman_asimi_sn: int) -> dict:
        """İşçiden bir protokol satırı okur.

        `readline` üzerinde doğrudan zaman aşımı yok; okuma ayrı bir iş
        parçacığında yapılıp burada sınırlanıyor. Aksi hâlde donmuş bir işçi
        API iş parçacığını süresiz kilitlerdi.
        """
        kutu: dict = {}

        def oku() -> None:
            try:
                kutu["satir"] = self._surec.stdout.readline()
            except Exception as exc:  # noqa: BLE001
                kutu["hata"] = exc

        parcacik = threading.Thread(target=oku, daemon=True)
        parcacik.start()
        parcacik.join(zaman_asimi_sn)

        if parcacik.is_alive():
            self._oldur()
            raise VlmHatasi(f"VLM {zaman_asimi_sn} sn içinde yanıt vermedi")
        if "hata" in kutu:
            self._oldur()
            raise VlmHatasi(f"VLM okunamadı: {kutu['hata']}")

        satir = (kutu.get("satir") or "").strip()
        if not satir:
            self._oldur()
            raise VlmHatasi("VLM süreci beklenmedik şekilde kapandı")
        try:
            return json.loads(satir)
        except json.JSONDecodeError as exc:
            raise VlmHatasi(f"VLM bozuk yanıt verdi: {satir[:120]}") from exc

    def _oldur(self) -> None:
        if self._surec is not None:
            try:
                self._surec.kill()
            except Exception:  # noqa: BLE001
                pass
            self._surec = None
        self._yukleme_sn = None

    def kapat(self) -> None:
        """İşçiyi düzgünce kapatır ve VRAM'i bırakır."""
        with self._kilit:
            self._zamanlayiciyi_iptal()
            if not self.acik:
                self._surec = None
                return
            try:
                self._surec.stdin.write(json.dumps({"kapat": True}) + "\n")
                self._surec.stdin.flush()
                self._surec.wait(timeout=15)
            except Exception:  # noqa: BLE001
                self._oldur()
            finally:
                self._surec = None
                self._yukleme_sn = None

    # --- boşta kapatma ---

    def _zamanlayiciyi_iptal(self) -> None:
        if self._zamanlayici is not None:
            self._zamanlayici.cancel()
            self._zamanlayici = None

    def _zamanlayiciyi_kur(self) -> None:
        self._zamanlayiciyi_iptal()
        self._zamanlayici = threading.Timer(self._bosta_sinir, self._bosta_kontrol)
        self._zamanlayici.daemon = True
        self._zamanlayici.start()

    def _bosta_kontrol(self) -> None:
        # Zamanlayıcı kurulduktan sonra yeni istek gelmiş olabilir; kapatmadan
        # önce gerçekten boşta mıyız diye bakılıyor.
        if not self.acik:
            return
        if time.monotonic() - self._son_kullanim >= self._bosta_sinir:
            self.kapat()
        else:
            with self._kilit:
                self._zamanlayiciyi_kur()

    # --- kullanım ---

    def _istek_gonder(self, govde: dict, zaman_asimi_sn: int) -> dict:
        """İşçiye bir istek yazıp yanıtını okur. Kilidi kendisi alır."""
        with self._kilit:
            if not self.acik:
                self._baslat()

            # Yanıt gibi istek de saf ASCII gidiyor: yol Türkçe karakter
            # içerirse (kullanıcı adı, klasör) aynı kodlama tuzağına düşerdi.
            try:
                self._surec.stdin.write(json.dumps(govde) + "\n")
                self._surec.stdin.flush()
            except Exception as exc:  # noqa: BLE001
                self._oldur()
                raise VlmHatasi(f"VLM'e istek yazılamadı: {exc}") from exc

            yanit = self._satir_oku(zaman_asimi_sn)
            self._son_kullanim = time.monotonic()
            self._zamanlayiciyi_kur()

        if not yanit.get("ok"):
            raise VlmHatasi(yanit.get("hata", "VLM isteği başarısız"))
        return yanit

    def oznitelik_cikar(self, foto_yolu: Path) -> dict:
        """Fotoğraftan öznitelik sözlüğü döner. Hata durumunda VlmHatasi atar."""
        yanit = self._istek_gonder(
            {"foto": str(foto_yolu.resolve())}, ISTEK_ZAMAN_ASIMI_SN
        )
        return yanit["oznitelik"]

    def oznitelik_cikar_coklu(self, foto_yolu: Path) -> list[dict]:
        """Fotoğraftaki bütün ürünleri kutularıyla döner.

        Zaman aşımı daha geniş: çoklu kipte model ürün başına ayrı bir nesne
        üretiyor ve dört ürünlü bir fotoğrafta ölçüm 22 saniye gösterdi (tek
        üründe 5 sn). Kalabalık bir raf fotoğrafı bunun katı olabilir.
        """
        yanit = self._istek_gonder(
            {"foto": str(foto_yolu.resolve()), "coklu": True},
            ISTEK_ZAMAN_ASIMI_SN_COKLU,
        )
        return yanit["urunler"]
