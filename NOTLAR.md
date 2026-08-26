# Rafino — Notlar

Defter (`STAJ-DEFTERI.md`) *ne yaptığımızı* tutuyor.
Bu dosya *henüz yapmadıklarımızı ve sonra dönülecek yerleri* tutuyor.

---

## 1. Sonra tekrar, daha iyi anlatılacak

### 1.1 Ölçümün kararları değiştirme hikâyesi — RAPORUN OMURGASI

Burak bunun daha iyi anlatılmasını istedi. Rapor yazarken bu bölüm öne alınacak;
kod satır sayısı veya özellik listesi değil.

Anlatılacak üç vaka — üçü de aynı kalıpta: **gerekçeli bir karar, ölçümle çürüdü.**

| # | Karar | Varsayım | Ölçüm | Sonuç |
|---|---|---|---|---|
| 1 | **K4 → K10** renk kaynağı | "Renk pikselden daha güvenilir çıkar" | piksel %38 / VLM %70; ayrışan 20 üründe VLM 20-0 | Karar tersine çevrildi |
| 2 | **Marka ground truth** | "Ürün adının ilk kelimesi markadır" | `Lino Perros` → `Lino`, `Peter England` → `Peter`; 45 sorgunun bir kısmı anlamsızdı | Değerlendirme setinin kendisi bozukmuş |
| 3 | **K13** kopya eşiği | "0.92 iyi bir sınır" | %50 küçültülmüş fotoğraf 0.896 veriyor | 0.88'e indirildi, sessiz kaçak önlendi |

Anlatırken vurgulanacak asıl fikir:
**Her üçünde de kararın gerekçesi mantıklıydı. Yanlış olan gerekçe değil,
gerekçenin ölçülmemiş olmasıydı.** Projeyi ödevden mühendisliğe çıkaran şey bu.

Ek olarak anlatılacak: eval setinin kendi denetimi (sandalet vakası) —
model yanlış sanılıyordu, ground truth bozuk çıktı.

### 1.2 Ölçüm kavramları

Aşağıdakiler bir daha, daha somut örneklerle anlatılacak:

- **Recall@1 / Recall@5** ve çok cevaplı sorgu tuzağı
  (`kategori R@1 = 0.117` düşük değil; 6 doğru cevap varken tavan zaten 0.167)
- **MRR** — Recall@1'in yalanını nasıl yakalıyor
- **Zor negatif (hard negative)** — neden rastgele değil aynı kategoriden seçildi
- **Kesinlik (precision) vs Kapsam (coverage)** — marka ölçümünde ayırdığımız şey
  ve bu ayrımın K11 kararını nasıl belirlediği
- **Eşik (threshold)** — dağılımların ayrık olup olmamasına göre nasıl seçilir

---

## 2. Model envanteri — hangisi ne işe yarıyor

| Model | Sorduğu soru | Rol | Durum |
|---|---|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | Bu fotoğraf neye benziyor? | İndeks A | ✅ Çalışıyor |
| `intfloat/multilingual-e5-large` | Bu metin neye benziyor? | İndeks B | ✅ Çalışıyor |
| `Qwen/Qwen3-VL-4B-Instruct` | Bu ürün **ne**? (kategori/marka/renk) | Kayıt hattı | ✅ Çalışıyor |
| `google/owlv2-base-patch16-ensemble` | Fotoğrafta kaç ürün var, **nerede**? | Çoklu ürün | ⚠️ Test edildi, **bağlanmadı** |
| SQLite FTS5 | Bu kelime hangi kayıtta geçiyor? | İndeks C | ✅ Çalışıyor |

**OWLv2 bir "algı" modeli değil, bir konum modeli.** VLM "bu ne?" sorusunu
cevaplıyor, OWLv2 "kaç tane ve nerede?" sorusunu. İkisi farklı iş.

---

## 3. Yapılacaklar

### Kısa vade (öncelikli)

- [ ] **FastAPI uçları + arayüz.** Sistemin tamamı şu an sadece terminalden
      kullanılabiliyor. Hocaya ve şirkete gösterilecek ekran yok.
      **Kalan en büyük ve en riskli parça** — arayüz işleri tahmini aşma
      eğilimindedir ("tahminini 2 ile çarp" kuralı en çok burada geçerli).
- [ ] Kopya tespitini kayıt hattına bağlamak (`src/ingest/kopya.py` hazır,
      çağıran yok).

### Orta vade

- [ ] **Barkod okuma (`pyzbar`).** `urun_kodu` sütunu şemada boş duruyor.
      Barkod, İndeks C'nin en güçlü olduğu veri türü: kesin, nadir, benzersiz.
      K11'in (marka kaynağı) tamamlayıcısı — okunabilen barkod, VLM tahminini ezer.
      Tahmini süre: yarım gün.
- [ ] **OWLv2'yi hatta bağlamak.** Şu an sistem "tek fotoğraf = tek ürün"
      varsayıyor. Gümrük fotoğrafları kolaj hâlinde geliyor ve VLM 15 üründen
      birini anlatıp 14'ünü **sessizce** yok sayıyor. Ölçülen maliyet düşük:
      0.62 GB VRAM, 0.62 sn/fotoğraf. Kutular arası örtüşme için NMS gerekecek.
- [ ] `ayirt_edici` alanının prompt kalitesi. Kemer için
      `"kemerli, kuma, kemerli"` gibi tekrarlı, anlamsız çıktılar üretiyor.

### Uzun vade — şimdi YAPILMAYACAK, gerekçeleriyle

- [ ] **Türkçe gövdeleyici (stemmer) — İndeks C için.**
      Sorun ölçüldü: `"kemer"` → 12 sonuç, `"kemerler"` → **0**, `"kemeri"` → **0**.
      Önek araması (`kemer*`) sadece ileri çalışıyor; kullanıcı çekimli hâli
      yazarsa kök bulunamıyor. Türkçe sondan eklemeli olduğu için İngilizceden
      daha çok acıtıyor.
      **Neden şimdi değil:** Hibritte İndeks B bu boşluğu kapatıyor —
      `kemerler` sorgusunda C sıfır dönerken B anlamsal olarak buluyor, RRF de
      boş dönen indeksi yok sayıyor. Üç indeksin birlikte olmasının somut faydası
      tam olarak bu.
      **Ne zaman yapılmalı:** stok 500+ ürüne çıkıp B'nin kurtarma oranı düşerse.
      Şimdi yapmak erken optimizasyon. (Snowball veya Zemberek entegrasyonu, 1-2 gün.)

- [ ] **Sorgu tipine göre ağırlıklı RRF.**
      Şu an üç indeks eşit ağırlıklı. Her indeks her sorgu tipinde iyi değil,
      dolayısıyla sabit ağırlık iyi olduğu yerde kazandırırken kötü olduğu yerde
      kaybettiriyor.
      **Neden şimdi değil:** K11 uygulandıktan sonra hibritte bozulan sorgu
      sayısı 11'den **2'ye** düştü (`mavi günlük ayakkabı`, `siyah güneş gözlüğü`).
      Kalan zarar çok küçük; ağırlık ayarı 2 sorgu için karmaşıklık eklemek olur.

- [ ] **Embedding LoRA fine-tune (Faz 7) — İPTAL.**
      Hibrit yapı hedefi zaten tutturdu (genel R@5 0.670 → 0.890).
      Contrastive eğitim büyük batch ister, 2 Eylül'e sığmaz, kazancı belirsiz.
      Yeniden gündeme gelirse: base 4-bit + compute dtype **bf16** (fp16 taşma
      yapıyor), Colab L4 üzerinde.

---

## 4. Kapanmış soru: "daha büyük VLM marka sorununu çözer mi?"

**Hayır. Ölçüldü.** `eval/kiyas_vlm.py` ile Qwen3-VL-4B ve 8B aynı 59 fotoğrafta
karşılaştırıldı.

| Alan | 4B (taban) | 8B | Fark |
|---|---:|---:|---|
| marka | 25/59 | **24/59** | **-1** |
| malzeme | 56/59 | 58/59 | +2 |
| kategori / renk / durum | 59/59 | 59/59 | — |

Marka detayı: 1 kayıt doldu (`Being human`), **2 kayıt kayboldu**
(Wrangler, adidas), 5 kayıtta çelişti. Çelişenlerin ikisinde 8B haklı —
`CAG -> GAS` ve `OBRAHU -> TITAN`, ikisi de 4B'nin uydurmasıydı.

**Sonuç:** 8B kesinliği bir miktar artırıyor ama **kapsamı hiç açmıyor.**
Fotoğrafta yazı okunmuyorsa daha büyük model de okuyamıyor. Yani marka
kapsamı sorunu bir model kapasitesi sorunu DEĞİL.

**Karar sonucu:** 4B'de kalınıyor (8B iki kat VRAM ve süre, karşılığı yok).
Ve K11 (marka elle giriş) bir geçici çözüm değil, **asıl çözüm** —
daha büyük model beklemenin faydası yok.

---

## 5. Kaldığımız yer — 20 Ağustos

### Tamamlanan

- **Tanıtım sitesi** (`tanitim/`) — çalışıyor, erişilebilirlik denetiminden geçti.
  Tasarım yönü ui-ux-pro-max verisinden alındı: Swiss/minimal + Lexend & Source Sans 3
  ("Corporate Trust" eşleşmesi, açıklaması "government, healthcare, accessibility").
  **Burak bu tasarımı beğendi ve ana uygulamaya taşınmasını istedi.**
- **Tasarım token sistemi** (`tokens.css`) — üç katman, tek ham renk kodu yok.
  Hem tanıtımda hem uygulamada aynı set kullanılıyor; fark yoğunlukta.
- **API katmanı** (`src/api/main.py`) — 13 uç: arama, ürün CRUD, stok hareketi,
  markası eksik listesi, uzun iş kalıbı.
- **Uygulama arayüzü** (`static/`) — `index.html`, `stil.css`, `uygulama.js`.
  Arama, sonuç kartları, detay paneli, düzeltme formu, stok hareketi formu,
  silme onayı, hareket geçmişi.

### YARIN İLK İŞ: doğrulama

API'de bir hata bulundu ve düzeltildi ama **çalışır hâlde doğrulanmadı**:

```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be
used in that same thread.
```

Sebep: FastAPI eşzamansız olmayan uçları iş parçacığı havuzunda çalıştırıyor,
her istek farklı parçacığa düşüyor; tek paylaşılan SQLite bağlantısı çalışmıyor.
Çözüm: `_baglanti()` artık parçacık başına bağlantı tutuyor (`threading.local`).
Derleme kontrolünden geçti, **çalışma zamanında test edilmedi.**

Doğrulama adımları:

1. `preview_start` ile `rafino` sunucusunu başlat (port 8000, model yüklemesi ~25 sn)
2. `curl http://127.0.0.1:8000/api/ozet` → 200 dönmeli
3. Arayüzde arama yap, ürüne tıkla, stok hareketi ekle, silme onayını dene

### Sonra: tanıtım sitesini 21st.dev ile yeniden yapmak

**Kısıt:** 21st.dev ücretsiz katman **günde 2 bileşen çekimi** veriyor
(`get_component` ücretli, `search` bedava ve sınırsız).

İlk arama sonuçları brief'e pek uymadı — çoğunda gradyan, plazma arka plan,
glassmorphism, koyu tema var. Brief ise "kamu kurumu, sade, neon yok".
En yakın aday: `HeroSection – Enterprise-Ready Landing Page Hero with Dual CTAs`
(id 8156, erişilebilirlik ve shadcn/ui vurgusu var).

Karar verilmesi gereken: 21st bileşenleri React+shadcn+Tailwind. Ya
(a) tanıtım sitesi React'e taşınır (derleme zinciri gelir), ya
(b) 2 çekim referans olarak kullanılıp mevcut token sistemine uyarlanır.
Burak'a sorulacak.

---

## 6. 20 Ağustos — düzeltilen üç hata ve VLM hattı

### 6.1 İş parçacığı hatası (düzeltildi, doğrulandı)

`sqlite3.ProgrammingError: SQLite objects created in a thread can only be used
in that same thread.` FastAPI eşzamansız olmayan uçları bir havuzda çalıştırıyor.

İki yerde vardı:
- `api/main.py` `_baglanti()` — tek paylaşılan bağlantı
- `search/arama.py` `Arayici._sqlite` — yaşam döngüsünde kurulup istekten kullanılıyordu

İkincisi daha sinsiydi: **indeks C her gerçek aramada 500 veriyordu.** Isınma
sorgusu ana parçacıkta çalıştığı için açılışta fark edilmiyordu.

Çözüm: her ikisinde de iş parçacığı başına bağlantı (`threading.local`).

### 6.2 Türkçe karakter (düzeltildi)

**Kodlama hatası yoktu.** Veritabanı, API ve sayfa baştan beri düzgün UTF-8.

Gerçek sebep: `prompts/oznitelik_renkli.txt` içindeki **kelime listeleri** ASCII
yazılmıştı (`gomlek`, `kirmizi`, `el cantasi`). Model listeden kopyalarken
harfleri de kopyalıyor; kendi cümlesini kurarken düzgün Türkçe yazıyordu. Sonuç
aynı cümlede karışık: `"yesil cam parfum ... koyu yeşil içeriği"`.

Yapılanlar:
- İstemdeki tüm listeler düzgün Türkçeye çevrildi + açık "harfleri aynen
  kopyala" kuralı eklendi
- `scripts/turkce_duzelt.py` ile mevcut 46 kayıt çevrildi (yedek alındı:
  `data/stok.20260820-170221.yedek.db`)
- `aciklama.py`'deki `durum != "saglam"` karşılaştırması sadeleştirmeli hâle
  getirildi — yoksa istem değişince HER ürün sessizce "hasarlı" sayılacaktı

### 6.3 `ı` harfi ve FTS — en önemli bulgu

`sqlite.py`'nin eski notu "`remove_diacritics 2` ı→i yapıyor" diyordu. **Yanlış.**

Unicode ayrışımı:
```
'ş' -> [LATIN SMALL LETTER S, COMBINING CEDILLA]     -> katlanıyor
'ğ' -> [LATIN SMALL LETTER G, COMBINING BREVE]       -> katlanıyor
'ı' -> [LATIN SMALL LETTER DOTLESS I]                -> KATLANMIYOR
```

`ı` başlı başına bir taban harf, atılacak aksanı yok. Türkçe klavyesi olmayan
operatör "kirmizi" yazınca `kırmızı` kaydını **bulamıyordu** — üstelik `ı` bu
alanın sözlüğünde en sık harf: kırmızı, sarı, ayakkabı, çantası, sırt.

Çözüm: `sqlite.tr_katla()` — indekse yazılan metin de sorgu da aynı katlamadan
geçiyor. Tek giriş noktası `sqlite.fts_yaz()`; `stok.guncelle` ve `sqlite.ekle`
oradan geçiyor. Gösterilen metin `urun` tablosunda düzgün Türkçe kalıyor.

Doğrulandı: kırmızı/kirmizi/KIRMIZI, gömlek/gomlek, ayakkabı/ayakkabi,
çantası/cantasi — hepsi aynı sonucu veriyor.

### 6.4 Ürün ekleme + VLM hattı (çalışıyor, ölçüldü)

Burak'ın itirazı yerindeydi: operatöre kategori/marka/renk yazdırmak, projeyi
fotoğraflı bir Excel'e indirger. Bkz. hafıza: `operator-yazmaz-dogrular`.

Kurulan mimari — **sıcak işçi süreç**:

| dosya | işi |
|---|---|
| `src/ingest/vlm.py` | ortak çıkarım mantığı (istem, çözünürlük, JSON ayıklama) |
| `scripts/vlm_isci.py` | modeli açık tutan uzun ömürlü süreç, stdin/stdout JSON protokolü |
| `src/ingest/vlm_servis.py` | işçiyi doğuran/kapatan yönetici, 10 dk boşta kapanma |
| `POST /api/is/oznitelik` | uzun iş kalıbıyla çıkarım başlatır |
| `GET /api/vlm/durum` | işçi açık mı, ne kadar boşta |

Neden bu mimari: model yüklemesi 69-110 sn, fotoğraf işleme 4-5 sn. Süreç başına
yükleme → her üründe 2 dk bekleme. API içine almak → VRAM kalıcı dolu (GPU 8 GB,
embedder'lar zaten yüklü). Sıcak işçi ikisinin arası: öbeğin ilk fotoğrafı
yükleme bedelini öder, gerisi saniyeler, öbek bitince VRAM geri döner.

Arayüz: fotoğraf yüklenince çıkarım kendiliğinden başlıyor, form dolu geliyor,
VLM'den gelen alanlar "fotoğraftan" rozetiyle işaretleniyor (operatör neyi
doğrulayacağını bilsin), alana dokununca rozet kalkıyor.

**Ölçüldü (20 Ağustos akşamı):**

| | süre | çıktı |
|---|---|---|
| ilk fotoğraf | 42.8 sn | model yükleniyor + çıkarım |
| sonraki | 4.9 sn | sıcak yol |
| sonraki | 5.0 sn | sıcak yol |

Çıktılar düzgün Türkçe: `gömlek`, `kumaş`, `sağlam`, `çizgili desenli`.
API üzerinden uçtan uca da doğrulandı: fotoğraf yükle → VLM → kayıt → üç
indekste de 1. sırada bulunuyor.

VRAM: embedder'lar 6367 MiB, VLM yerleşik ~1.6 GB ekliyor (896 piksel sınırı
sayesinde eski nottaki 3.9 GB değil). Çıkarım anında tepe 7932/8188 MiB'a
çıkıyor — sığıyor ama dar; boşta kapatma bu yüzden önemli.

**Yakalanan hata:** çocuk süreç Windows'ta stdout'u cp1254 ile kodluyordu,
ebeveyn UTF-8 okuyordu; Türkçe içeren her yanıt `'utf-8' codec can't decode
byte 0xf6` ile düşüyor ve yönetici süreci öldürüp modeli yeniden yüklüyordu.
Protokol saf ASCII'ye (`ensure_ascii=True`) çevrildi.

### 6.5 Barkod — bilinçli olarak boş

Arayüzde yeri açıldı, içi doldurulmadı. Sebep kayıtlı: şirketin hangi barkod
sistemini kullandığı bilinmiyor (EAN-13 / Code 128 / QR / özel format farklı
çözümler gerektiriyor). Şimdilik numara "Ürün kodu" alanına elle yazılabiliyor.

### 6.6 Doğrulanan diğer şeyler

- İş kuralı 400 + Türkçe mesaj: "Stokta 1 adet var, 5 adet çıkış yapılamaz"
- Eksik ürün 404, yol geçişi (`../config.py`) engelleniyor
- Stok hareketi uçtan uca: adet 1→3→1, hareket geçmişi doğru
- `scripts/test_stok.py`: 23/23 geçiyor

---

## 7. Arama alakası — "bavul" sorunu (çözüldü)

**Şikâyet:** "bavul" yazınca alakasız 20 ürün geliyordu.

**Sebep:** Vektör araması "bulamadım" diyemez — her zaman en yakın k komşuyu
döndürür, ne kadar uzak olursa olsun. Depoda olmayan bir şey arandığında bu,
alakasız ürünlerden ikna edici bir liste üretiyordu. Sistemin bilmediğini
bilmemesi, yanlış cevap vermesinden beter.

**Ölçüm (61 kayıt, en yakın komşu kosinüs mesafesi):**

|          | depoda VAR | depoda YOK | karar |
|---|---|---|---|
| İndeks A (görsel) | 0.594 – 0.722 | 0.806 – 0.864 | rahat ayrışıyor → eşik **0.78** |
| İndeks B (metin)  | 0.127 – 0.218 | 0.154 – 0.211 | **BANTLAR ÇAKIŞIYOR** |

İndeks B'de depodaki "Lino Perros" 0.218 alırken depoda olmayan "kahve fincanı"
0.154 alıyor. Yani B'yi tek başına kapı bekçisi yapacak bir eşik matematiksel
olarak yok. e5 sıralamada iyi, "bu bizde var mı" sorusunda değil.

**Çözüm:** Her indeksi güvenilir olduğu yerde kullanmak.
- A: eşik 0.78 (net ayrışma)
- B: eşik 0.147 — yalnızca gerçekten ayrıştığı bölge. Elediklerini A veya C
  yakalıyor: mouse, kemer, steelseries, Lino Perros tam olarak böyle bulunuyor.
- C: `OR` yerine `AND`. `"kahve"*` öneki "kahverengi"ye tutunduğu için
  "kahve makinesi" sorgusu sekiz güneş gözlüğü döndürüyordu. İndeks C'nin işi
  kesinlik; geri çağırmayı A ve B sağlıyor.

**Sonuç: 25/25.** 13 "var olması beklenen" sorgu doğru ilk sonuçla geliyor,
12 "olmayan" sorgu boş dönüyor. "kahverengi çanta" hâlâ çalışıyor (7 sonuç),
yani önek araması topyekûn kırılmadı.

### Arayüzden kaldırılan: A·B·C rozetleri

Hangi indeksin bulduğu geliştirme bilgisi; depocunun işine yaramıyor ve arayüzü
kalabalıklaştırıyordu. API yanıtında `siralar` alanı DURUYOR — ölçüm ve hata
ayıklama için gerekli, yalnızca ekrana basılmıyor.

### 7.1 RRF ağırlıkları — "steelserie" vakası

Burak "steelserie" yazdığında mouse 2. çıkıyordu:

```
1. STEELE gri saat          skor=0.01639  {gorsel: 1}   <- görsel gürültü
2. steelseries Beyaz mouse  skor=0.01639  {fts: 1}      <- tam marka öneki
```

Skorlar **birebir eşit**; beraberlik keyfî bozulup yanlış kayıt öne geçiyordu.
Eşit ağırlık, indeks C'nin kesin marka eşleşmesini indeks A'nın rastgele görsel
benzerliğiyle aynı kefeye koyuyordu — oysa indeks C'nin var oluş sebebi tam da
gömme modellerinin markalarda başarısız olması.

13 sorgudan oluşan etiketli kümede ağırlık taraması:

| fts ağırlığı | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
|---|---|---|---|---|---|---|
| top-1 doğru | 12/13 | 13/13 | 13/13 | 13/13 | 13/13 | 13/13 |

`RRF_AGIRLIKLARI = {"gorsel": 1.0, "metin": 1.0, "fts": 2.0}` — 1.5 sınırının
belirgin üstünde ama C'yi tek başına hâkim kılmıyor. Kontrol: "STEELE" araması
hâlâ saati getiriyor, yani aşırı düzeltme yok.

### 7.2 Tanıtım sitesinde düzeltilen yanlış iddia

Sayfa "Barkod desteği"ni **şimdiki zamanda** anlatıyordu: "Fotoğrafta barkod
okunabiliyorsa ... ürün kodu alanı bu veriyle dolar." Böyle bir özellik yok.
Değerlendiricinin okuyacağı bir sayfada olmayan özelliği var göstermek, ölçülmüş
gerçek sonuçların güvenilirliğini de düşürür. "Barkod için hazırlık [PLANLANAN]"
olarak yeniden yazıldı ve neden bekletildiği açıklandı.

Tanıtım sitesi bunun dışında dün bıraktığımız hâlde; `.claude/launch.json`'a
önizleme yapılandırması eklendi.

## 8. Önbellek tuzağı (çözüldü)

`StaticFiles` varsayılan hâliyle tarayıcının eski `uygulama.js`'i kullanmasına
izin veriyordu. Bu, "kod doğru ama arayüz güncellenmiyor" gibi görünen ama
aslında var olmayan hatalar aratıyordu — hem bana hem Burak'a. Artık statik
dosyalar `Cache-Control: no-cache` ile veriliyor: dosya her kullanımdan önce
sunucuya soruluyor, değişmemişse 304 dönüyor.


## 9. Yazı tipleri yerele alındı

**Sorun ölçüldü, tahmin edilmedi:**

| kaynak | süre |
|---|---|
| `tokens.css` (yerel) | 16 ms |
| `stil.css` (yerel) | 33 ms |
| `uygulama.js` (yerel) | 60 ms |
| **Google Fonts CSS** | **746 ms** |

`<link rel="stylesheet">` render'ı bloke ediyor. Zamanlama bunu kanıtladı:
Google isteği 1133 ms'de bitiyor, `domInteractive` 1137 ms'de gerçekleşiyordu —
sayfa doğrudan bu isteği bekliyordu.

Üç sebep daha:
- İnternet yokken Segoe UI'ye düşüyor ve metin %10 daralıyor (aynı dizgede
  142.4 px yerine 129.1 px) — yerleşim kayıyor, tasarım ölçüleri Lexend'e göre.
- Kurumsal ağ paketi *reddetmek* yerine sessizce *düşürürse* tarayıcı TCP zaman
  aşımını bekler; sayfa onlarca saniye boş kalabilir. Depo için gerçekçi senaryo.
- Her açılışta operatörün IP'si Google'a gidiyor (KVKK).

**Türkçe için kritik ayrım:**

```
ç ö ü ı  →  latin      (U+0000-00FF + U+0131 açıkça eklenmiş)
ğ ş      →  latin-ext  (U+0100-02BA)
```

İki alt küme de indirilmeli. Yalnızca `latin` alınırsa "ağırlık" kelimesinde
ğ harfi kelime ortasında başka yazı tipine düşer. `@font-face` kuralları
`unicode-range` ile yazıldığı için tarayıcı gerekeni kendi seçiyor.

**Sonuç (aynı sayfa, aynı makine):**

| | önce | sonra |
|---|---|---|
| dış kaynak isteği | 1 (746 ms) | **0** |
| `domInteractive` | 1137 ms | **346 ms** |
| `loadEventEnd` | 1611 ms | **369 ms** |

Tanıtım sitesinde `domInteractive` 19 ms.

`scripts/fontlari_indir.py` ile indiriliyor (tekrar çalıştırılabilir, ağırlık
eklemek gerekirse elle uğraşmaya gerek yok). 10 dosya, 391 KB; `static/fonts/`
ve `tanitim/fonts/` altına aynı set yazılıyor — iki site bilerek ayrı duruyor,
paylaşılan bir dizin bağımlılık kurardı.

Yan düzeltme: `stil.css`'te `font-weight: 700` vardı ama yüklenen ağırlıklar
400/500/600. Tarayıcı sahte kalın üretiyordu; 600'e çekildi.

## 10. Çoklu ürün modeli — tek fotoğrafta birden çok ürün

Depoda ürünler çoğu zaman tek tek değil, raf/koli hâlinde fotoğraflanıyor. Tek
ürün kipi böyle bir fotoğrafta yalnızca en büyük nesneyi anlatıp gerisini
sessizce atıyordu.

**Çözüm:** VLM'den sınırlayıcı kutu da isteniyor (`prompts/oznitelik_coklu.txt`),
her ürün kırpılıp AYRI kaydediliyor.

Kritik tasarım noktası: **kimlik kırpığın içerik özetinden üretiliyor.** Tüm
fotoğrafın özeti kullanılsaydı aynı fotoğraftaki üç ürün aynı kimliği alır ve
indeks A'da üçü de aynı vektörle temsil edilirdi — görsel arama onları
ayıramazdı. Kırpınca hem kimlikler ayrışıyor hem her ürün kendi görseliyle
gömülüyor.

**Ölçüm (4 ürünlü kompozit, 20 Ağustos):**
- 4 ürünün 4'ü de doğru kategori + isabetli kutuyla bulundu (gözle doğrulandı)
- Belirteç sınırı 200→900'e çıkarıldı: 4 ürün 368 belirteç tutuyor, 200'de
  çıktı ortadan kesilip JSON hiç ayrıştırılamıyordu

**Beklenmedik kazanç — marka okuma:** İki aşamalı yapıldı. 1. aşama kutuları
bulur, 2. aşama her kırpığa AYRI sorar. Ölçümde tüm fotoğraftan markaların 0/4'ü
okundu, kırpıklardan 1/4 — ve o tek marka gözle okunabilen tek markaydı
("Being human"). Kırpınca ürüne düşen piksel artıyor. Bu, projenin en zayıf
metriğini (marka %43) doğrudan iyileştiriyor. Bedel: ürün başına ~4.4 sn ek.

## 11. Güvenlik denetimi (20 maddelik liste + aktif sızma)

İzole bir sunucuda (ayrı port, geçici veritabanı) gerçek saldırılarla test
edildi. Senin verine hiç dokunulmadı.

**Sağlam çıkanlar:**

| # | madde | sonuç |
|---|---|---|
| 6 | SQL injection | `DROP TABLE`, `' OR '1'='1` zararsız string; tüm dinamik SQL ya `?` ya beyaz liste |
| 7 | Sunucu doğrulama | Pydantic: geçersiz tip/negatif miktar → 422 |
| 8 | Ham HTML render | Tüm alanlarda XSS payload'ı `kacir()` ile kaçırıldı; tarayıcıda `alert` tetiklenmedi |
| 4 | Frontend izinleri / path traversal | `Path(x).name` mutlak yol, backslash, çift kodlamayı da eziyor |
| 12 | CORS `*` | CORS ara katmanı yok (varsayılan kapalı) |
| 14 | Tahmin edilebilir ID | `sha256(içerik)[:16]`, ardışık değil |
| 15 | Body'yi direkt kaydetmek | İzin dışı alanlar (`durum`, `adet`) sessizce atılıyor |
| 16 | İmzasız webhook | Webhook yok |
| 17 | Production stack trace | Hatalar temiz mesaj; debug/reload kapalı |
| 18 | Eski dependency | FastAPI 0.141, Pydantic 2.13, Pillow 12.2 — güncel |
| 20 | Dosya upload validation | Uzantı + içerik (`Image.verify()`) + 25 MB; `.php`, sahte `.jpg`, SVG, 50 MB reddedildi |
| 1,2,9,13,19 | Şifre/auth ile ilgili | Uygulamada auth katmanı YOK — bu maddeler konu dışı |

**Bulunan ve düzeltilenler:**

- **Miktar üst sınırı yoktu.** 10^18 giriş kabul ediliyordu; tek daktilo hatası
  stok kaydını mahvediyordu. `Field(le=1_000_000)` eklendi.
- **Metin alanları sınırsızdı.** `max_length=200/500` eklendi (marka, kategori,
  açıklama vb.).
- **OpenAPI docs açıktı (madde 11).** `/docs`, `/redoc`, `/openapi.json` tüm uç
  envanterini sızdırıyordu. `RAFINO_URETIM=1` ile kapanıyor; geliştirmede açık.
- **Eşzamanlı iş sınırı yoktu (madde 5).** VLM işleri kuyruğu sınırsız
  büyüyebiliyordu. `_AZAMI_ESZAMANLI_IS=4`, aşınca 429.

**Bilinçli olarak yapılmayanlar:**

- **Rate limiting (madde 5):** Genel HTTP rate limiter EKLENMEDİ. Uygulama
  127.0.0.1'e bağlı, tek kullanıcılı yerel bir araç. Asıl korunması gereken
  pahalı kaynak GPU ve onu eşzamanlı iş sayacı koruyor.
- **Kimlik doğrulama (madde 1,2,9,13,19):** Auth yok. Tek kullanıcılı yerel
  araç için makul. **AMA:** dağıtım senaryosu değişirse (çok kullanıcı, ağa
  açık), auth + gerçek rate limiter ŞART — çünkü şu an ağa erişen herkes tüm
  stoku okuyabilir/değiştirebilir/silebilir. Bu, gelecekteki en büyük risk.

## 12. Açık konular

- **Marka kapsamı %43 → çoklu-ürün hattı bunu iyileştiriyor.** Kırpıp yeniden
  sorma marka okumayı artırıyor (bkz. bölüm 10). Tek-ürün ekleme akışına da aynı
  "kırp ve yeniden sor" adımı eklenebilir; henüz eklenmedi.
- **Kimlik doğrulama yok.** Dağıtım ağa açılırsa en büyük risk (bkz. bölüm 11).
- **Google Fonts yerine yerel** (bölüm 9) tamamlandı; internet olmadan test
  edildi.
- **Çoklu ürün için arayüz yok.** API ucu (`/api/is/coklu-oznitelik`) ve VLM
  hattı hazır ve ölçüldü, ama `static/` tarafında henüz "birden çok ürün bulundu,
  hepsini onayla" ekranı yok. Sıradaki iş.

## 13. Açık riskler ve kayıtlı uyarılar

- **Kopya eşiği örneklemi küçük.** `AYNI_URUN_ESIGI = 0.88` yalnızca **10 zor
  negatif çift** üzerinden belirlendi. Stoğa aynı modelin farklı renkleri
  eklendiğinde negatiflerin üst sınırı yükselebilir ve eşik yukarı çekilmesi
  gerekebilir. `scripts/test_kopya.py` tekrar çalıştırılmalı.

- **Değerlendirme seti moda ağırlıklı.** Kaynak veri seti giyim/aksesuar;
  gümrük deposunda kablo, elektronik, kozmetik, gıda da var. Ölçülen sayılar
  bu alan için üst sınır sayılmamalı.

- **Marka kapsamı %43.** VLM ürünlerin ancak %43'ünde marka okuyabiliyor.
  K11 (elle giriş) bunu telafi ediyor ama arayüz olmadan telafi çalışmıyor —
  yani arayüz gecikirse bu ölçülen kazanç kâğıt üzerinde kalır.

- **Ürün adı FTS5'e KONULMADI** ve konulmamalı. O ground truth; indekslenirse
  ölçüm anlamsızlaşır. İleride biri "arama zayıf" diye ekleme eğilimine girerse
  bu not hatırlatsın.
