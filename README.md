# Rafino

**Görsel tabanlı akıllı stok takibi.** Fotoğrafını yükle, Türkçe sor, rafını öğren.

Ürün fotoğrafı yüklenir; sistem ürünü tanır, özniteliklerini çıkarır ve indeksler.
Kullanıcı ürün kodunu bilmeden, günlük Türkçe ile arar ("mavi valiz") ve ilgili ürünü
raf konumuyla birlikte bulur.

Okul stajı projesi · Başlangıç 10 Ağustos 2026 · Teslim 2 Eylül 2026

Günlük ilerleme ve alınan kararların gerekçeleri: [STAJ-DEFTERI.md](STAJ-DEFTERI.md)

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
| Çoklu ürün (opsiyonel) | `google/owlv2-base-patch16-ensemble` | Apache-2.0, kutu döndürür |

Renk modelden sorulmaz, pikselden hesaplanır (HSV + k-means → sabit palet).
Sebep: VLM aynı çağrıda alan ve serbest metin isteyince kendi içinde çelişiyordu.

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

## Donanım

RTX 4070 Laptop — 8 GB VRAM. **İki model asla aynı anda belleğe yüklenmez.**
Kayıt ve sorgu farklı zamanlarda, farklı modellerle çalışır.

## Klasör yapısı

```
data/photos/    ürün fotoğrafları (git'e dahil — yeniden üretilemez)
data/chroma/    vektör veritabanı (git'te yok — yeniden üretilebilir)
docs/           teknik referans
eval/           değerlendirme seti ve ölçüm betikleri
prompts/        VLM promptları (koda gömülmez, dosyadan okunur)
scripts/        yardımcı betikler
src/            uygulama kodu
static/         arayüz
```
