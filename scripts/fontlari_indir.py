"""Google Fonts'taki yazı tiplerini indirip yerele alır.

Neden yerele alınıyor (20 Ağustos ölçümü):

- `<link rel="stylesheet">` render'ı bloke ediyor ve Google'dan cevap 746 ms
  sürüyordu. Yerel dosyalarımızın toplamı 109 ms. `domInteractive` 1137 ms'de
  gerçekleşiyordu, Google isteği 1133 ms'de bitiyordu — yani sayfa doğrudan bu
  isteği bekliyordu.
- İnternet yokken Segoe UI'ye düşüyor ve metin %10 daralıyor (aynı dizgede
  142.4 px yerine 129.1 px): yerleşim kayıyor.
- Kurumsal ağ paketi reddetmek yerine sessizce düşürürse tarayıcı TCP zaman
  aşımını bekliyor ve sayfa onlarca saniye boş kalabiliyor. Depo ortamı için
  gerçekçi senaryo bu.
- Her açılışta operatörün IP'si Google'a gidiyor.

TÜRKÇE İÇİN KRİTİK: `ç ö ü` temel `latin` alt kümesinde ama `ğ ı ş` ayrı bir
dosyada, `latin-ext` içinde. Yalnızca `latin` indirilirse "ağırlık" gibi
kelimelerde harfler kelime ortasında başka yazı tipine düşer. Bu yüzden iki alt
küme de indiriliyor ve `@font-face` kuralları `unicode-range` ile yazılıyor —
tarayıcı hangi dosyaya ihtiyaç duyduğunu kendisi seçiyor.

Kullanım:
    python scripts/fontlari_indir.py
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

# Google, User-Agent'a bakarak farklı biçimler sunuyor: eski bir UA ile woff2
# yerine ttf geliyor ve dosyalar üç katına çıkıyor. Modern bir UA şart.
KULLANICI_ARACISI = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

CSS_ADRESI = (
    "https://fonts.googleapis.com/css2"
    "?family=Lexend:wght@400;500;600"
    "&family=Source+Sans+3:wght@400;600"
    "&display=swap"
)

# Yalnızca bu ikisi indiriliyor. Kiril, Yunan ve Vietnamca alt kümeleri
# Google'ın CSS'inde var ama bu uygulamada hiç kullanılmıyor.
GEREKLI_ALT_KUMELER = {"latin", "latin-ext"}

# Arayüzün kopyaları ayrı klasörlerde duruyor; ikisine de yazılıyor.
HEDEFLER = [config.PROJE_KOK / "static", config.PROJE_KOK / "tanitim"]


def _getir(adres: str) -> bytes:
    istek = urllib.request.Request(adres, headers={"User-Agent": KULLANICI_ARACISI})
    with urllib.request.urlopen(istek, timeout=60) as yanit:
        return yanit.read()


def _bloklari_ayikla(css: str) -> list[dict]:
    """CSS'ten @font-face bloklarını sökerek alt küme/ağırlık/URL çıkarır."""
    bloklar = []
    for esles in re.finditer(
        r"/\*\s*([a-z-]+)\s*\*/\s*@font-face\s*\{([^}]*)\}", css
    ):
        alt_kume, govde = esles.group(1), esles.group(2)
        if alt_kume not in GEREKLI_ALT_KUMELER:
            continue
        aile = re.search(r"font-family:\s*'([^']+)'", govde)
        agirlik = re.search(r"font-weight:\s*(\d+)", govde)
        url = re.search(r"url\(([^)]+)\)", govde)
        aralik = re.search(r"unicode-range:\s*([^;]+);", govde)
        if not (aile and agirlik and url and aralik):
            continue
        bloklar.append({
            "aile": aile.group(1),
            "agirlik": agirlik.group(1),
            "alt_kume": alt_kume,
            "url": url.group(1),
            "aralik": aralik.group(1).strip(),
        })
    return bloklar


def _dosya_adi(blok: dict) -> str:
    kisa = blok["aile"].lower().replace(" ", "-")
    return f"{kisa}-{blok['agirlik']}-{blok['alt_kume']}.woff2"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("[i] Google Fonts CSS alınıyor...")
    css = _getir(CSS_ADRESI).decode("utf-8")
    bloklar = _bloklari_ayikla(css)
    if not bloklar:
        print("[!] Hiç @font-face bloğu ayrıştırılamadı. CSS biçimi değişmiş olabilir.")
        return 1

    bulunan = {b["alt_kume"] for b in bloklar}
    if "latin-ext" not in bulunan:
        # Sessizce devam etmek, Türkçe harflerin bozuk görüneceği bir sürüm
        # üretirdi; erken ve gürültülü durmak doğru.
        print("[!] latin-ext alt kümesi yok — ğ, ı, ş harfleri eksik kalırdı.")
        return 1

    print(f"[i] {len(bloklar)} dosya indirilecek ({', '.join(sorted(bulunan))})\n")

    icerikler: dict[str, bytes] = {}
    kurallar: list[str] = []
    toplam = 0
    for blok in sorted(bloklar, key=lambda b: (b["aile"], b["agirlik"], b["alt_kume"])):
        ad = _dosya_adi(blok)
        if ad not in icerikler:
            icerikler[ad] = _getir(blok["url"])
            kb = len(icerikler[ad]) / 1024
            toplam += kb
            print(f"  {ad:<42} {kb:6.1f} KB")
        kurallar.append(
            "@font-face {\n"
            f"  font-family: '{blok['aile']}';\n"
            "  font-style: normal;\n"
            f"  font-weight: {blok['agirlik']};\n"
            "  font-display: swap;\n"
            f"  src: url('fonts/{ad}') format('woff2');\n"
            f"  unicode-range: {blok['aralik']};\n"
            "}"
        )

    baslik = (
        "/* Yerele alınmış yazı tipleri — scripts/fontlari_indir.py ile üretildi.\n"
        " * ELLE DÜZENLEMEYİN; betiği yeniden çalıştırın.\n"
        " *\n"
        " * Google Fonts'tan çekmek sayfayı 746 ms bloke ediyordu ve internet\n"
        " * yokken tipografi Segoe UI'ye düşüp yerleşimi kaydırıyordu.\n"
        " *\n"
        " * `unicode-range` kuralları KORUNDU: Türkçe'de ç, ö, ü temel `latin`\n"
        " * alt kümesinde ama ğ, ı, ş `latin-ext` içinde. Tarayıcı hangi dosyaya\n"
        " * ihtiyacı olduğuna bu aralıklara bakarak karar veriyor.\n"
        " */\n\n"
    )
    css_metni = baslik + "\n\n".join(kurallar) + "\n"

    for hedef in HEDEFLER:
        font_dizini = hedef / "fonts"
        font_dizini.mkdir(parents=True, exist_ok=True)
        for ad, veri in icerikler.items():
            (font_dizini / ad).write_bytes(veri)
        (hedef / "fonts.css").write_text(css_metni, encoding="utf-8")
        print(f"\n[+] {hedef.name}/fonts/ ({len(icerikler)} dosya) + {hedef.name}/fonts.css")

    print(f"\n[i] Toplam {toplam:.1f} KB")
    print("[!] HTML'lerden Google Fonts <link> satırları kaldırılmalı ve")
    print('    <link rel="stylesheet" href="fonts.css"> eklenmeli.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
