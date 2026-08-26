"""VLM ile öznitelik çıkarımının ortak parçaları.

Hem toplu betik (`scripts/vlm_toplu.py`) hem sıcak işçi (`scripts/vlm_isci.py`)
buradan besleniyor. Ayrı kopyalar tutulsaydı istem sürümü, çözünürlük sınırı ve
JSON ayıklama kuralları zamanla birbirinden kayardı; ikisi farklı sonuç üreten
iki hat olurdu ve hangisinin ölçülmüş olduğu belirsizleşirdi.

Model bu modülde YÜKLENMİYOR — yükleme çağıranın işi. Sebep VRAM: 8 GB'lık bir
dizüstü GPU'sunda VLM (4-bit ~3.9 GB) ile arama embedder'ları rahat rahat yan
yana durmuyor. Kimin ne zaman yükleyip bırakacağına süreç sahibi karar veriyor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src import config

# Qwen3-VL dinamik çözünürlük kullanıyor: görsel büyüdükçe görü token sayısı,
# VRAM ve süre birlikte artıyor. 1800x2400 fotoğraflarda VRAM 3.9 GB yerine
# 7.2 GB'a çıkıyor. Öznitelik çıkarımı için bu çözünürlük gereksiz; uzun kenar
# 896 piksel etiket okumaya fazlasıyla yetiyor.
AZAMI_KENAR = 896

# Üretilen belirteç sınırı. Beklenen çıktı tek satırlık bir JSON; 200 belirteç
# fazlasıyla yeterli ve model saçmalamaya başlarsa erken kesiyor.
AZAMI_BELIRTEC = 200

# Çoklu ürün çıkarımında her ürün için ayrı bir nesne + sınırlayıcı kutu
# üretiliyor. 200 belirteçle dört ürünlü bir fotoğrafta çıktı ortasından
# kesiliyor ve JSON hiç ayrıştırılamıyordu; ölçümde dört ürün 368 belirteç
# tuttu, 900 rahat pay bırakıyor.
AZAMI_BELIRTEC_COKLU = 900

VARSAYILAN_ISTEM = "oznitelik_renkli.txt"
VARSAYILAN_ISTEM_COKLU = "oznitelik_coklu.txt"

# Kutuları biraz genişletiyoruz: model nesneyi sıkı çevreliyor ve kenardaki
# kayış, sap, gölge kırpılmış görüntünün dışında kalabiliyor. Ölçümde saatin
# alt kordonu tam sınırdaydı.
KUTU_PAYI = 0.03


def istem_oku(ad: str = VARSAYILAN_ISTEM) -> str:
    return (config.PROMPT_DIZINI / ad).read_text(encoding="utf-8")


def json_ayikla(ham: str) -> dict | None:
    """Modelin ham çıktısından JSON söker.

    Model istem ne kadar net olursa olsun bazen ```json ile sarıyor ya da
    öncesine bir cümle ekliyor. Üç aşama deneniyor: doğrudan, çitler atılarak,
    son çare ilk `{...}` bloğu. Hiçbiri tutmazsa None — çağıran bunu hata
    olarak sayar, uydurulmuş bir sözlük döndürmek çok daha kötü olurdu.
    """
    temiz = re.sub(r"^```(?:json)?\s*", "", ham.strip())
    temiz = re.sub(r"\s*```$", "", temiz)
    try:
        return json.loads(temiz)
    except json.JSONDecodeError:
        pass
    esles = re.search(r"\{.*\}", temiz, re.DOTALL)
    if esles:
        try:
            return json.loads(esles.group(0))
        except json.JSONDecodeError:
            return None
    return None


def model_yukle(model_id: str | None = None):
    """VLM'i 4-bit NF4 nicemlemeyle yükler. (islemci, model) döner.

    Yükleme 69-110 saniye sürüyor; çağıran bunu bir kez ödeyip modeli açık
    tutmalı. Fotoğraf başına yeniden yüklemek, işlemenin kendisinden (4-5 sn)
    yirmi kat pahalı.
    """
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    nicemleme = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kimlik = model_id or config.VLM_MODEL
    islemci = AutoProcessor.from_pretrained(kimlik)
    model = AutoModelForImageTextToText.from_pretrained(
        kimlik,
        quantization_config=nicemleme,
        device_map={"": 0},
        dtype=torch.bfloat16,
    ).eval()
    return islemci, model


def _sor(islemci, model, istem: str, foto_yolu: Path, azami_belirtec: int) -> str:
    """Fotoğraf + istemi modele verip ham metni döner."""
    import torch
    from PIL import Image

    gorsel = Image.open(foto_yolu).convert("RGB")
    gorsel.thumbnail((AZAMI_KENAR, AZAMI_KENAR), Image.Resampling.LANCZOS)

    mesaj = [{
        "role": "user",
        "content": [
            {"type": "image", "image": gorsel},
            {"type": "text", "text": istem},
        ],
    }]
    girdi = islemci.apply_chat_template(
        mesaj, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        cikti = model.generate(
            **girdi, max_new_tokens=azami_belirtec, do_sample=False
        )
    return islemci.decode(
        cikti[0][girdi["input_ids"].shape[1]:], skip_special_tokens=True
    )


def oznitelik_cikar(islemci, model, istem: str, foto_yolu: Path) -> dict | None:
    """Tek fotoğraftan öznitelik sözlüğü çıkarır. Ayrıştırılamazsa None."""
    return json_ayikla(_sor(islemci, model, istem, foto_yolu, AZAMI_BELIRTEC))


def oznitelik_cikar_coklu(
    islemci, model, istem: str, foto_yolu: Path
) -> list[dict] | None:
    """Fotoğraftaki BÜTÜN ürünleri kutularıyla birlikte döner.

    Her öğede `kutu` alanı var: [sol, üst, sağ, alt], 0-1000 aralığına
    ölçeklenmiş. Kutusuz veya bozuk kutulu öğeler ELENİYOR — kutu olmadan
    kırpma yapılamaz ve kırpma olmadan ürünlerin görsel vektörleri birbirinden
    ayrışmaz (hepsi aynı fotoğrafı gömerdi).
    """
    ham = _sor(islemci, model, istem, foto_yolu, AZAMI_BELIRTEC_COKLU)
    veri = json_ayikla(ham)
    if veri is None:
        return None

    urunler = veri.get("urunler") if isinstance(veri, dict) else veri
    if not isinstance(urunler, list):
        return None

    gecerli = []
    for u in urunler:
        if not isinstance(u, dict):
            continue
        kutu = u.get("kutu")
        if (isinstance(kutu, list) and len(kutu) == 4
                and all(isinstance(v, (int, float)) for v in kutu)
                and kutu[0] < kutu[2] and kutu[1] < kutu[3]):
            gecerli.append(u)
    return gecerli


def kutuyu_kirp(foto_yolu: Path, kutu: list, pay: float = KUTU_PAYI):
    """0-1000 ölçeğindeki kutuyu piksele çevirip fotoğrafı kırpar.

    Kırpılmış görüntü hem kaydın fotoğrafı hem de kimliğinin kaynağı oluyor:
    kimlik kırpığın içerik özetinden üretildiği için aynı fotoğraftaki iki ürün
    farklı kimlik alıyor ve indeks A'da ayrı vektörlerle temsil ediliyor.
    """
    from PIL import Image

    gorsel = Image.open(foto_yolu).convert("RGB")
    G, Y = gorsel.size
    x1, y1, x2, y2 = (float(v) for v in kutu)

    # Pay kutunun kendi boyutuna oranlı; küçük nesnede küçük, büyükte büyük.
    dx, dy = (x2 - x1) * pay, (y2 - y1) * pay
    kutu_px = (
        max(0, int((x1 - dx) / 1000 * G)),
        max(0, int((y1 - dy) / 1000 * Y)),
        min(G, int((x2 + dx) / 1000 * G)),
        min(Y, int((y2 + dy) / 1000 * Y)),
    )
    return gorsel.crop(kutu_px)
