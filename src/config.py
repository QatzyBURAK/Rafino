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

# --- Alaka eşikleri ---
# Vektör araması HER ZAMAN en yakın k komşuyu döndürür, ne kadar uzak olursa
# olsun. Depoda olmayan bir şey arandığında ("bavul") bu, alakasız ürünlerden
# kendinden emin bir liste üretiyordu — kullanıcı için en kötü davranış, çünkü
# sistem bilmediğini bilmiyormuş gibi görünüyor.
#
# 20 Ağustos ölçümü (61 kayıt), en yakın komşunun kosinüs mesafesi:
#
#            depoda VAR olan sorgular   depoda OLMAYAN sorgular
#   İndeks A   0.594 – 0.722              0.806 – 0.864
#   İndeks B   0.127 – 0.218              0.154 – 0.211
#
# İndeks A rahat ayrışıyor: aradaki boşluğa 0.78 konuyor.
#
# İndeks B'de bantlar ÇAKIŞIYOR. Depodaki "Lino Perros" 0.218 alırken depoda
# olmayan "kahve fincanı" 0.154 alıyor — yani B'yi tek başına kapı bekçisi
# yapacak bir eşik yok. e5 sıralamada iyi, "bu bizde var mı" sorusunda değil.
#
# Bu yüzden B'nin eşiği gerçekten ayrıştığı yere çekildi (0.147). Bunun altında
# kalan sorgular güvenle kabul ediliyor; üstünde kalanları B kabul etmiyor ama
# kayıt yine de A veya C'den geçebiliyor. Ölçümdeki dört "VAR" sorgusu (mouse,
# kemer, steelseries, Lino Perros) tam olarak böyle bulunuyor: B eliyor, marka
# ve kategori eşleşmesi sayesinde C alıyor.
#
# Somut kazanç: "kahve makinesi" sorgusu artık boş dönüyor. Eskiden `"kahve"*`
# öneki "kahverengi"ye tutunduğu ve B de yakın bulduğu için üç güneş gözlüğü
# geliyordu.
#
# DİKKAT: bu sayılar 61 kayıtlık kataloğa göre. Katalog büyüdükçe mesafe
# dağılımı kayar; eval/ altındaki ölçümle yeniden bakılmalı.
GORSEL_ALAKA_ESIGI = 0.78
METIN_ALAKA_ESIGI = 0.147

# --- RRF ağırlıkları ---
# Eşit ağırlıkta birleştirme, kesin bir eşleşmeyi rastgele bir benzerlikle aynı
# kefeye koyuyordu. "steelserie" sorgusunda:
#
#   STEELE gri saat          skor=0.01639  {gorsel: 1}   <- görsel gürültü
#   steelseries Beyaz mouse  skor=0.01639  {fts: 1}      <- tam marka öneki
#
# Skorlar birebir eşit çıkıyor ve beraberlik keyfî bozulup yanlış kayıt 1.
# oluyordu. Oysa indeks C'nin var oluş sebebi tam da bu: gömme modelleri nadir
# belirteçlerde (marka, ürün kodu) kötü, anahtar kelime araması ise tam
# eşleşmeyi kesin biliyor. AND'e geçtikten sonra C yalnızca gerçek eşleşmede
# konuşuyor, yani konuştuğunda ona daha çok kulak vermek doğru.
#
# 13 sorgudan oluşan etiketli küme üzerinde ağırlık taraması (20 Ağustos):
#
#   fts ağırlığı   1.0    1.5    2.0    2.5    3.0    4.0
#   top-1 doğru    12/13  13/13  13/13  13/13  13/13  13/13
#
# 1.5'ten itibaren düzeliyor ve 4.0'a kadar hiçbir sorgu bozulmuyor. 2.0
# seçildi: sınırın belirgin üstünde ama C'yi tek başına hâkim kılmıyor.
RRF_AGIRLIKLARI = {"gorsel": 1.0, "metin": 1.0, "fts": 2.0}

# --- Dosya türleri ---
RESIM_UZANTILARI = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dizinleri_hazirla() -> None:
    """Eksik veri dizinlerini oluşturur. Her giriş noktasında çağrılabilir."""
    for d in (VERI_DIZINI, FOTO_DIZINI, CHROMA_DIZINI, PROMPT_DIZINI, EVAL_DIZINI):
        d.mkdir(parents=True, exist_ok=True)
