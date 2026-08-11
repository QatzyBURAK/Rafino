"""Tüm yollar, model kimlikleri ve sabitler tek yerde.

Kural: hiçbir modülde elle yazılmış yol veya model kimliği bulunmaz, hepsi buradan gelir.
Yollar dosya konumundan türetilir, böylece hangi dizinden çalıştırıldığından bağımsız.
"""

from pathlib import Path

# --- Yollar ---
PROJE_KOK = Path(__file__).resolve().parent.parent

VERI_DIZINI = PROJE_KOK / "data"
FOTO_DIZINI = VERI_DIZINI / "photos"
CHROMA_DIZINI = VERI_DIZINI / "chroma"
SQLITE_YOLU = VERI_DIZINI / "stok.db"

PROMPT_DIZINI = PROJE_KOK / "prompts"
EVAL_DIZINI = PROJE_KOK / "eval"

# --- Modeller ---
# Görsel embedding — İndeks A. Fotoğraf ve metni aynı uzaya gömer.
GORSEL_MODEL = "Qwen/Qwen3-VL-Embedding-2B"

# Metin embedding — İndeks B. VLM'in ürettiği Türkçe açıklamayı gömer.
# Sıfat-isim bağını görsel modellerden iyi tuttuğu için "mavi valiz" burada çözülür.
METIN_MODEL = "intfloat/multilingual-e5-large"

# Öznitelik çıkarımı — kategori, marka, malzeme, durum. Renk BURADAN gelmez.
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

# --- Embedding ---
# Matryoshka: 2048 yerine 512. Küçük veride kalite aynı, arama ve disk daha ucuz.
GORSEL_BOYUT = 512

# multilingual-e5-large sabit 1024 boyut üretir, kırpılmaz.
METIN_BOYUT = 1024

# e5 ailesi bu önekleri bekler; koymazsan kalite belirgin düşer.
E5_BELGE_ONEKI = "passage: "
E5_SORGU_ONEKI = "query: "

# --- ChromaDB ---
GORSEL_KOLEKSIYON = "urun_gorsel"
METIN_KOLEKSIYON = "urun_metin"
MESAFE_METRIGI = "cosine"

# --- Arama ---
# RRF sabiti. Literatürdeki standart değer; tek bir indeksin ilk sıralarının
# sonucu tek başına ele geçirmesini engeller.
RRF_K = 60
VARSAYILAN_SONUC = 5

# --- Dosya türleri ---
RESIM_UZANTILARI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dizinleri_hazirla() -> None:
    """Eksik veri dizinlerini oluşturur. Her giriş noktasında çağrılabilir."""
    for d in (VERI_DIZINI, FOTO_DIZINI, CHROMA_DIZINI, PROMPT_DIZINI, EVAL_DIZINI):
        d.mkdir(parents=True, exist_ok=True)
