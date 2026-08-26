"""VLM'i açık tutan uzun ömürlü işçi süreç — arayüzden tek ürün eklemek için.

Neden ayrı bir süreç:
VLM yüklemesi 69-110 saniye, bir fotoğrafı işlemesi 4-5 saniye. Fotoğraf başına
süreç açmak yükleme bedelini her seferinde ödemek demek; operatör her üründe iki
dakika bekler. Modeli API sürecinin içine almak ise VRAM'i kalıcı tutar ve 8 GB'lık
bir GPU'da arama embedder'larıyla çakışır. Ayrı ve sürekli bir süreç ikisinin
arasını buluyor: yükleme bir kez ödeniyor, süreç kapandığında VRAM eksiksiz geri
dönüyor (işletim sistemi topluyor — `torch.cuda.empty_cache()` bunu tam yapmıyor).

Protokol — satır başına bir JSON, stdin'den istek, stdout'a yanıt:

    <- {"hazir": true, "yukleme_sn": 92.4}        (model yüklenince bir kez)
    -> {"foto": "C:\\...\\data\\photos\\abc.jpg"}
    <- {"ok": true, "oznitelik": {"kategori": "...", ...}}
    -> {"kapat": true}
    <- {"gule_gule": true}

stdout YALNIZCA protokol için: ilerleme ve hata metinleri stderr'e gidiyor. İkisi
karışsaydı çağıran tarafta ayrıştırma sessizce bozulurdu.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Günlükler (stderr) sunucu çıktısına akıyor; Windows'un cp1254'ü Türkçe
# harflerde patlıyor. Protokolün kendisi zaten saf ASCII (bkz. `_yaz`).
for _akis in (sys.stdout, sys.stderr):
    if hasattr(_akis, "reconfigure"):
        _akis.reconfigure(encoding="utf-8", errors="replace")

from src.ingest import vlm  # noqa: E402


def _yaz(nesne: dict) -> None:
    """Protokol satırı yazar.

    `ensure_ascii=True` (varsayılan) bilerek kullanılıyor: Türkçe harfler
    \\uXXXX olarak kaçırılıyor ve satır saf ASCII kalıyor. Windows'ta boruya
    bağlı stdout cp1254 ile kodlanıyor, ebeveyn ise UTF-8 bekliyor; ham Türkçe
    yazınca "'utf-8' codec can't decode byte 0xf6" ile her yanıt düşüyordu.
    Kaçırılmış hâl `json.loads` tarafında düzgün Türkçeye geri dönüyor.

    flush şart: stdout bir boruya bağlandığında Python tamponluyor ve yanıt
    çağırana ancak tampon dolunca ulaşıyor — yani hiç ulaşmıyor gibi görünüyor.
    """
    sys.stdout.write(json.dumps(nesne) + "\n")
    sys.stdout.flush()


def _bilgi(metin: str) -> None:
    print(metin, file=sys.stderr, flush=True)


def main() -> int:
    istem_adi = sys.argv[1] if len(sys.argv) > 1 else vlm.VARSAYILAN_ISTEM
    istem = vlm.istem_oku(istem_adi)
    # Çoklu istem ayrı dosyada ve ayrı belirteç sınırı istiyor; ikisi de baştan
    # okunuyor ki kip değişince model yeniden yüklenmesin.
    istem_coklu = vlm.istem_oku(vlm.VARSAYILAN_ISTEM_COKLU)

    _bilgi(f"[i] VLM yükleniyor (istem: {istem_adi})...")
    t0 = time.perf_counter()
    try:
        islemci, model = vlm.model_yukle()
    except Exception as exc:  # noqa: BLE001
        # Yükleme hatası (çoğunlukla VRAM yetersizliği) çağırana bildirilmeli;
        # sessizce ölmek, API tarafında zaman aşımı olarak görünürdü.
        _yaz({"hazir": False, "hata": f"{type(exc).__name__}: {exc}"})
        return 1
    yukleme = time.perf_counter() - t0
    _bilgi(f"[+] Yüklendi ({yukleme:.0f} sn). İstek bekleniyor.")
    _yaz({"hazir": True, "yukleme_sn": round(yukleme, 1)})

    for satir in sys.stdin:
        satir = satir.strip()
        if not satir:
            continue
        try:
            istek = json.loads(satir)
        except json.JSONDecodeError:
            _yaz({"ok": False, "hata": "Geçersiz istek satırı"})
            continue

        if istek.get("kapat"):
            _yaz({"gule_gule": True})
            return 0

        yol = Path(istek.get("foto", ""))
        if not yol.exists():
            _yaz({"ok": False, "hata": f"Fotoğraf bulunamadı: {yol.name}"})
            continue

        coklu = bool(istek.get("coklu"))
        t = time.perf_counter()
        try:
            if coklu:
                sonuc = vlm.oznitelik_cikar_coklu(islemci, model, istem_coklu, yol)
            else:
                sonuc = vlm.oznitelik_cikar(islemci, model, istem, yol)
        except Exception as exc:  # noqa: BLE001
            _yaz({"ok": False, "hata": f"{type(exc).__name__}: {exc}"})
            continue

        if sonuc is None:
            _yaz({"ok": False, "hata": "Model geçerli JSON döndürmedi"})
            continue

        sure = time.perf_counter() - t
        _bilgi(f"  {yol.name}{' [çoklu]' if coklu else ''}: {sure:.1f} sn")
        if coklu:
            _yaz({"ok": True, "urunler": sonuc, "sure_sn": round(sure, 1)})
        else:
            _yaz({"ok": True, "oznitelik": sonuc, "sure_sn": round(sure, 1)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
