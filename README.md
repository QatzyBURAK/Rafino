# Rafino

**Görsel tabanlı akıllı stok takibi.** Fotoğrafını yükle, Türkçe sor, rafını öğren.

Ürün fotoğrafı yüklenir; sistem ürünü tanır, özniteliklerini çıkarır ve indeksler.
Kullanıcı ürün kodunu bilmeden, günlük Türkçe ile arar ("mavi valiz") ve ilgili ürünü
raf konumuyla birlikte bulur.

---

## Problem

Depolarda ürün bilgisi elle giriliyor: yavaş ve hataya açık. Ürün çeşidinin fazla olduğu
her depoda aynı sorun var. Fikir, gümrük depolarındaki süreç incelenerek çıktı.

Sistem iki şeyi çözüyor:

1. **Elle veri girişi** — kategori, marka, renk ve durum fotoğraftan otomatik çıkarılır.
2. **Ürün kodu bilme zorunluluğu** — arama doğal dille yapılır.

## Temel mimari kararı

Bu bir **sınıflandırma değil, erişim (retrieval)** problemi. Depoya yarın ne geleceği
belli değil; sabit sınıflı bir model, listede olmayan ilk üründe çöker. Bunun yerine her
ürün bir vektöre dönüşür ve arama, vektör uzayında en yakın komşuyu bulmaktır. Yeni ürün
tipi geldiğinde model yeniden eğitilmez, sadece yeni vektör eklenir.

## Arama: üç indeks, tek sonuç

Tek bir vektör uzayı yetmiyor. Görsel-metin modelleri sıfat ile nesneyi birbirine bağlamakta
zayıf: "mavi valiz" sorgusunda üzerinde mavi etiket olan siyah valiz de yüksek skor alır.
Bu yüzden üç ayrı indeks kullanılıyor — **başarısızlık şekilleri örtüşmediği için**:

| İndeks | Ne saklar | Güçlü olduğu sorgu |
|---|---|---|
| A — görsel | Fotoğrafın vektörü | "şuna benzer ürün", kopya tespiti |
| B — metin | Öznitelik JSON'undan üretilen Türkçe cümlenin vektörü | "mavi valiz" |
| C — anahtar kelime | Marka ve ürün kodu (SQLite FTS5) | "Samsonite", "SM-A546B" |

Sonuçlar Reciprocal Rank Fusion ile birleştirilir: `skor = Σ 1/(60 + sıra)`.
Skorlar değil **sıralar** toplanır; farklı modellerin benzerlik değerleri farklı
ölçeklerde olduğu için doğrudan kıyaslanamaz.

## Model yığını

| Rol | Model | Not |
|---|---|---|
| Görsel embedding | `Qwen/Qwen3-VL-Embedding-2B` | Apache-2.0, `truncate_dim=512` |
| Öznitelik çıkarımı | `Qwen/Qwen3-VL-4B-Instruct` | 4-bit NF4, etiketten marka okur |
| Metin embedding | `intfloat/multilingual-e5-large` | MIT, Türkçe |
| Çoklu ürün | Öznitelik VLM'inin kendisi | Tek fotoğraftaki ürünleri sınırlayıcı kutularıyla döndürür |

Çoklu ürün başta ayrı bir konum modeliyle (OWLv2) planlanmıştı; VLM zaten kutu
üretebildiği için ayrı model yüklenmedi. Her kutu ayrı kırpılıp ayrı kaydediliyor
(kimlik kırpığın içeriğinden üretiliyor, yoksa aynı fotoğraftaki ürünler görsel
indekste ayrışmazdı).

**Renk kaynağı — ölçümle değişen bir karar.** Başta renk pikselden hesaplanıyordu
(HSV + k-means → sabit palet); varsayım "piksel modelden güvenilir" idi. Ölçüm
tersini gösterdi (VLM %70 / piksel %38 doğru; ayrışan 20 üründe VLM 20-0), karar
tersine çevrildi ve canlı yolda renk artık VLM'den alınıyor. Piksel yolu
(`src/ingest/renk.py`) ölçümde referans olarak duruyor.

## Kurulum

Sanal ortam `uv` ile yönetiliyor, Python 3.11.

```
uv venv .venv --python 3.11
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Kurulumdan sonra doğrula:

```
.venv\Scripts\python.exe scripts\check_env.py
```

Bu betik özellikle şunu yakalar: `torch.__version__` sonunda `+cpu` varsa GPU hiç
kullanılmıyordur ve her şey sessizce yavaş çalışır.

Birebir sürüm kilidi için `requirements.lock.txt` kullanılır (çalışan ortamdan
`pip freeze` ile üretildi; torch indeksi notu dosyanın başında).

## Çalıştırma

Web arayüzü, API ve veritabanı **tek komutla** ayağa kalkar. SQLite gömülüdür,
ayrı bir veritabanı sunucusu gerekmez; arayüz de FastAPI tarafından `/` altında
servis edilir.

```
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

`Application startup complete` satırını görünce (ilk açılışta arama modelleri
yüklenir, ~15-25 sn) tarayıcıda `http://localhost:8000` aç. Arayüz iki ana akış
sunar:

- **Arama** — ürünü günlük dille yaz ("mavi el çantası"); üç indeks + RRF sonucu
  raf konumuyla getirir. Stokta yoksa boş döner, alakasız ürün sıralamaz.
- **Ürün ekle** — fotoğrafı yükle; kategori, marka ve renk fotoğraftan otomatik
  dolar, operatör yalnızca doğrular/düzeltir. VLM ilk üründe yüklenir (~40 sn),
  sonraki ürünler saniyeler içinde.

`--reload` **kullanılmaz**: yeniden yükleme kipi modelleri her değişiklikte
baştan yükler ve VLM alt sürecini bozar.

Ağa açık bir kuruluma geçilirse (`--host 0.0.0.0`) önce kimlik doğrulama
eklenmelidir; şu an auth yok (bilinçli, tek kullanıcılı yerel araç). Üretimde
etkileşimli API dokümanlarını kapatmak için `RAFINO_URETIM=1` verilir.

## Değerlendirme verisi

Ölçüm için kullanılan ürün fotoğrafları
[`PestoRosso/lamoda-fashion-product-images`](https://huggingface.co/datasets/PestoRosso/lamoda-fashion-product-images)
veri setinden alınıyor (Kaggle *Fashion Product Images Dataset*'in yüksek
çözünürlüklü alt kümesi, MIT lisansı ile yayımlanmış).

**Fotoğraflar bu depoda tutulmuyor.** Veri setinin lisansı yükleyici tarafından
MIT olarak beyan edilmiş olsa da, altlarındaki ürün fotoğrafları e-ticaret
sitesine ve markalara ait; yeniden dağıtmak yerine yeniden üretiyoruz.

Seti kurmak için:

```
.venv\Scripts\python.exe scripts\veriseti_kur.py 60
```

Bu komut fotoğrafları `data/photos/` altına yazar, ground truth'u
`eval/urunler.jsonl` ve sorguları `eval/sorgular.jsonl` olarak üretir.
Ürün kimlikleri kayıtlı olduğu için sonuç her makinede birebir aynıdır.

Sıfırdan tam kurulum (üç indeksi de doldurmak) sırasıyla:

```
.venv\Scripts\python.exe scripts\veriseti_kur.py 60   # fotoğraflar + ground truth
.venv\Scripts\python.exe scripts\faz0.py              # İndeks A (görsel)
.venv\Scripts\python.exe scripts\vlm_toplu.py         # VLM öznitelikleri -> data/oznitelikler.jsonl
.venv\Scripts\python.exe scripts\indeks_b_kur.py vlm  # İndeks B (metin; renk kaynağı: vlm)
.venv\Scripts\python.exe scripts\indeks_c_kur.py      # İndeks C (SQLite FTS5)
```

Sıra önemli: `indeks_b_kur.py`, `vlm_toplu.py`'nin ürettiği `oznitelikler.jsonl`
dosyasını okur. Bu adımlardan sonra `uvicorn` ile arayüz açılabilir.

## Donanım

RTX 4070 Laptop — 8 GB VRAM. **İki model asla aynı anda belleğe yüklenmez.**
Kayıt ve sorgu farklı zamanlarda, farklı modellerle çalışır.

## Klasör yapısı

```
data/photos/    ürün fotoğrafları (git'te YOK — veriseti_kur.py ile üretilir)
data/chroma/    vektör veritabanı (git'te yok — indeks betikleriyle üretilir)
data/stok.db    SQLite: ürün, stok hareketi, İndeks C (git'te yok)
eval/           değerlendirme seti ve ölçüm betikleri
prompts/        VLM promptları (koda gömülmez, dosyadan okunur)
scripts/        yardımcı ve indeks-kurulum betikleri
src/            uygulama kodu (api, db, ingest, search, models)
static/         web arayüzü
tanitim/        tek sayfalık tanıtım sitesi
```
