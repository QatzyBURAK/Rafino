/* Rafino arayüzü — çerçevesiz, derleme adımı yok.
 *
 * Kamu kurumunda kullanım kriterleri koda yansıdı:
 *   - Her eylemin yazılı adı var; yalnız ikonla anlatılan işlem yok
 *   - Silme onay ister ve gerekçe sorar (gerekçe hareket kaydına girer)
 *   - Boş / yükleniyor / hata durumları ayrı ayrı ele alınıyor, hiçbiri
 *     sessizce boş ekran bırakmıyor
 *   - Sonucun hangi indeksten geldiği gösteriliyor; kapalı kutu arama
 *     kullanıcının sisteme güvenmesini zorlaştırıyor
 *   - Klavye: Escape paneli kapatır, "/" aramaya odaklanır
 */

'use strict';

const durum = {
  gorunum: 'arama',
  sorgu: '',
  secili: null,
  sonuclar: [],
};

const $ = (secici) => document.querySelector(secici);

/* ------------------------------------------------------------------ */
/* Yardımcılar                                                         */
/* ------------------------------------------------------------------ */

/** Metni HTML'e gömmeden önce kaçırır.
 *  Ürün adları ve marka bilgisi veritabanından geliyor; doğrudan innerHTML'e
 *  yazmak enjeksiyon riski oluşturur. */
function kacir(deger) {
  if (deger === null || deger === undefined) return '';
  return String(deger)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}

async function api(yol, secenekler = {}) {
  const yanit = await fetch(yol, {
    headers: { 'Content-Type': 'application/json' },
    ...secenekler,
  });
  if (!yanit.ok) {
    let mesaj = `Sunucu hatası (${yanit.status})`;
    try {
      const govde = await yanit.json();
      // FastAPI iş kuralı ihlallerini `detail` içinde döndürüyor
      // ("stokta 3 adet var, 5 adet çıkış yapılamaz" gibi). Bu mesajlar
      // kullanıcı için yazıldı, aynen gösteriliyor.
      if (govde.detail) mesaj = govde.detail;
    } catch { /* gövde JSON değilse varsayılan mesaj kalır */ }
    throw new Error(mesaj);
  }
  return yanit.json();
}

function bildir(metin, tur = 'bilgi') {
  const alan = $('#bildirimAlan');
  const kutu = document.createElement('div');
  kutu.className = `bildirim bildirim--${tur}`;
  kutu.textContent = metin;
  alan.appendChild(kutu);
  setTimeout(() => kutu.remove(), 4500);
}

function urunAdi(u) {
  // Ürünün insan tarafından okunabilir adı, elde olan alanlardan kuruluyor.
  const parcalar = [u.renk, u.kategori].filter(Boolean);
  let ad = parcalar.join(' ') || 'İsimsiz ürün';
  if (u.marka) ad = `${u.marka} ${ad}`;
  return ad.charAt(0).toUpperCase() + ad.slice(1);
}

const KAYNAK_ETIKET = {
  vlm: ['Modelden okundu', 'vlm'],
  elle: ['Elle girildi', 'elle'],
  barkod: ['Barkoddan', 'barkod'],
  bilinmiyor: ['Bilinmiyor', 'bilinmiyor'],
};

/* ------------------------------------------------------------------ */
/* Durum ekranları                                                     */
/* ------------------------------------------------------------------ */

function iskeletGoster(adet = 4) {
  $('#sonucAlani').innerHTML = Array.from({ length: adet })
    .map(() => '<div class="iskelet"></div>').join('');
  $('#sonucAlani').setAttribute('aria-busy', 'true');
}

function durumGoster({ baslik, metin, tur = '' }) {
  $('#sonucAlani').setAttribute('aria-busy', 'false');
  const ikon = tur === 'hata'
    ? '<path d="M24 14v14M24 34v.01" stroke="currentColor" stroke-width="3" stroke-linecap="round"/><circle cx="24" cy="24" r="20" stroke="currentColor" stroke-width="3"/>'
    : '<circle cx="21" cy="21" r="14" stroke="currentColor" stroke-width="3"/><path d="M32 32l10 10" stroke="currentColor" stroke-width="3" stroke-linecap="round"/>';
  $('#sonucAlani').innerHTML = `
    <div class="durum ${tur === 'hata' ? 'durum--hata' : ''}">
      <svg width="48" height="48" viewBox="0 0 48 48" fill="none" aria-hidden="true">${ikon}</svg>
      <h3>${kacir(baslik)}</h3>
      <p>${kacir(metin)}</p>
    </div>`;
}

/* ------------------------------------------------------------------ */
/* Liste                                                               */
/* ------------------------------------------------------------------ */

function kartHtml(u) {
  const seciliMi = durum.secili === u.kimlik;
  const foto = u.dosya
    ? `<img class="urun-foto" src="/api/foto/${encodeURIComponent(u.dosya)}" alt="" loading="lazy">`
    : `<div class="urun-foto urun-foto--yok"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true"><rect x="3" y="4" width="14" height="12" rx="1.5" stroke="currentColor" stroke-width="1.4"/><path d="M3 13l4-4 3 3 3-3 4 4" stroke="currentColor" stroke-width="1.4"/></svg></div>`;

  const detaylar = [u.kategori, u.marka || 'markası bilinmiyor'].filter(Boolean)
    .map(kacir).join(' <span class="ayrac">·</span> ');

  // A/B/C indeks rozetleri kaldırıldı: hangi indeksin bulduğu geliştirme
  // bilgisi, depocunun işine yaramıyor ve arayüzü kalabalıklaştırıyordu.
  // `siralar` alanı API yanıtında duruyor — ölçüm ve hata ayıklama için hâlâ
  // gerekli, yalnızca ekrana basılmıyor.

  const adetSinif = u.adet === 0 ? 'adet-etiket adet-etiket--tukendi' : 'adet-etiket';

  return `
    <button class="urun-kart" data-kimlik="${kacir(u.kimlik)}" aria-current="${seciliMi}">
      ${foto}
      <div class="urun-bilgi">
        <div class="urun-ad">${kacir(urunAdi(u))}</div>
        <div class="urun-detay">${detaylar}</div>
      </div>
      <div class="urun-sag">
        <span class="${adetSinif}">${u.adet ?? 0} ad.</span>
        ${u.raf ? `<span class="raf-etiket">${kacir(u.raf)}</span>` : ''}
      </div>
    </button>`;
}

function listeCiz(urunler) {
  $('#sonucAlani').setAttribute('aria-busy', 'false');
  if (!urunler.length) {
    durumGoster({
      baslik: durum.gorunum === 'arama' ? 'Sonuç bulunamadı' : 'Bu listede ürün yok',
      metin: durum.gorunum === 'arama'
        ? 'Farklı kelimelerle deneyin. Renk ve ürün türünü birlikte yazmak genelde daha iyi sonuç verir.'
        : 'Kayıt eklendiğinde burada görünecek.',
    });
    return;
  }
  $('#sonucAlani').innerHTML =
    `<div class="kart-liste">${urunler.map(kartHtml).join('')}</div>`;
}

/* ------------------------------------------------------------------ */
/* Veri yükleme                                                        */
/* ------------------------------------------------------------------ */

async function ozetYukle() {
  try {
    const o = await api('/api/ozet');
    $('#ozetCesit').textContent = o.cesit ?? 0;
    $('#ozetAdet').textContent = o.toplam_adet ?? 0;
    $('#ozetEksik').textContent = o.markasi_eksik ?? 0;
    $('#sayacTumu').textContent = o.cesit ?? 0;
    $('#sayacEksik').textContent = o.markasi_eksik || '';
  } catch (hata) {
    // Özet başarısız olursa sayfa yine çalışmalı; sadece sayılar boş kalır.
    console.warn('Özet yüklenemedi:', hata.message);
  }
}

async function aramaYap(sorgu) {
  durum.sorgu = sorgu;
  $('#listeBaslik').textContent = 'Arama sonuçları';
  if (!sorgu.trim()) {
    $('#listeAdet').textContent = '';
    durumGoster({
      baslik: 'Aramaya başlayın',
      metin: 'Aradığınız ürünü günlük dille yazın. Ürün kodu bilmenize gerek yok.',
    });
    return;
  }
  iskeletGoster();
  try {
    const veri = await api(`/api/ara?q=${encodeURIComponent(sorgu)}&k=20`);
    durum.sonuclar = veri.sonuclar;

    // Hiçbir indeks yeterince yakın bir kayıt bulamadıysa boş dönüyor.
    // Eskiden vektör araması en yakın 20 komşuyu her hâlükârda döndürdüğü
    // için "bavul" araması alakasız ürünlerden dolu bir liste veriyordu;
    // sistemin bilmediğini bilmemesi, yanlış cevap vermesinden beterdi.
    if (veri.adet === 0) {
      $('#listeAdet').textContent = '';
      durumGoster({
        baslik: `“${sorgu}” için kayıt bulunamadı`,
        metin: 'Bu tarife uyan bir ürün stokta görünmüyor. '
             + 'Farklı bir kelimeyle deneyebilir veya ürünü fotoğrafıyla ekleyebilirsiniz.',
      });
      return;
    }

    $('#listeAdet').textContent = `${veri.adet} sonuç`;
    listeCiz(veri.sonuclar);
  } catch (hata) {
    $('#listeAdet').textContent = '';
    durumGoster({ baslik: 'Arama başarısız', metin: hata.message, tur: 'hata' });
  }
}

async function listeYukle(eksikMarka) {
  $('#listeBaslik').textContent = eksikMarka ? 'Markası eksik kayıtlar' : 'Tüm ürünler';
  iskeletGoster();
  try {
    const veri = await api(`/api/urunler?limit=100${eksikMarka ? '&eksik_marka=true' : ''}`);
    durum.sonuclar = veri.urunler;
    $('#listeAdet').textContent = `${veri.toplam} kayıt`;
    listeCiz(veri.urunler);
  } catch (hata) {
    $('#listeAdet').textContent = '';
    durumGoster({ baslik: 'Liste yüklenemedi', metin: hata.message, tur: 'hata' });
  }
}

function gorunumDegistir(yeni) {
  durum.gorunum = yeni;
  document.querySelectorAll('.yan-dugme').forEach((d) => {
    d.setAttribute('aria-current', String(d.dataset.gorunum === yeni));
  });
  // Ekleme görünümü listeyi tamamen değiştiriyor; arama kutusu ve sonuç
  // alanı gizleniyor ki ekranda iki ayrı "ana içerik" durmasın.
  const ekleme = yeni === 'ekle';
  $('#aramaAlani').style.display = yeni === 'arama' ? '' : 'none';
  $('#ekleAlani').hidden = !ekleme;
  $('#listeBasligiSatir').hidden = ekleme;
  $('#sonucAlani').hidden = ekleme;

  if (ekleme) { kategorileriDoldur(); return; }
  if (yeni === 'arama') aramaYap($('#aramaGirdi').value);
  else listeYukle(yeni === 'eksik');
}

/* ------------------------------------------------------------------ */
/* Detay paneli                                                        */
/* ------------------------------------------------------------------ */

const HAREKET_ADI = {
  giris: ['Giriş', '+', 'giris'],
  cikis: ['Çıkış', '−', 'cikis'],
  duzeltme: ['Sayım düzeltmesi', '=', 'duzeltme'],
  silme: ['Stoktan çıkarıldı', '×', 'silme'],
};

function panelHtml(u) {
  const [kaynakAd, kaynakSinif] = KAYNAK_ETIKET[u.marka_kaynagi] || KAYNAK_ETIKET.bilinmiyor;

  const markaSatiri = u.marka
    ? `${kacir(u.marka)}<span class="kaynak-etiket kaynak-etiket--${kaynakSinif}">${kaynakAd}</span>`
    : `<span class="alan-deger--bos">bilinmiyor</span><span class="kaynak-etiket kaynak-etiket--bilinmiyor">Tamamlanmalı</span>`;

  const hareketler = (u.hareketler || []).slice().reverse().map((h) => {
    const [ad, isaret, sinif] = HAREKET_ADI[h.tip] || [h.tip, '?', 'duzeltme'];
    const tarih = new Date(h.tarih).toLocaleString('tr-TR',
      { day: '2-digit', month: '2-digit', year: '2-digit', hour: '2-digit', minute: '2-digit' });
    return `
      <li class="hareket">
        <span class="hareket-ikon hareket-ikon--${sinif}" aria-hidden="true">${isaret}</span>
        <span>
          <span class="hareket-ad">${ad}</span>
          <span class="hareket-not">· ${kacir(tarih)}</span>
          ${h.aciklama ? `<br><span class="hareket-not">${kacir(h.aciklama)}</span>` : ''}
        </span>
        <span class="hareket-sayi">${h.onceki} → ${h.sonraki}</span>
      </li>`;
  }).join('');

  const silinmis = u.durum === 'silindi';

  return `
    ${silinmis ? `
      <div class="uyari uyari--dikkat">
        <svg width="15" height="15" viewBox="0 0 15 15" fill="none" aria-hidden="true">
          <path d="M7.5 2l6 11h-12l6-11z" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/>
          <path d="M7.5 6v3M7.5 11v.01" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/>
        </svg>
        <span>Bu ürün stoktan çıkarılmış. Kaydı ve geçmişi saklanıyor ama aramada görünmüyor.</span>
      </div>` : ''}

    ${u.dosya ? `<img src="/api/foto/${encodeURIComponent(u.dosya)}" alt="${kacir(urunAdi(u))} fotoğrafı"
       style="width:100%;border-radius:var(--yuvarlak);border:1px solid var(--kenarlik);margin-bottom:var(--a-4)">` : ''}

    <div class="panel-bolum">
      <h3>Öznitelikler</h3>
      <div class="alan"><span class="alan-ad">Kategori</span><span class="alan-deger">${kacir(u.kategori) || '<span class="alan-deger--bos">—</span>'}</span></div>
      <div class="alan"><span class="alan-ad">Marka</span><span class="alan-deger">${markaSatiri}</span></div>
      <div class="alan"><span class="alan-ad">Renk</span><span class="alan-deger">${kacir(u.renk) || '<span class="alan-deger--bos">—</span>'}</span></div>
      <div class="alan"><span class="alan-ad">Ürün kodu</span><span class="alan-deger">${kacir(u.urun_kodu) || '<span class="alan-deger--bos">girilmemiş</span>'}</span></div>
      <div class="alan"><span class="alan-ad">Raf</span><span class="alan-deger">${kacir(u.raf) || '<span class="alan-deger--bos">—</span>'}</span></div>
      <div class="alan"><span class="alan-ad">Adet</span><span class="alan-deger"><strong>${u.adet}</strong></span></div>
    </div>

    ${silinmis ? '' : `
    <div class="panel-bolum">
      <h3>Bilgileri düzelt</h3>
      <form id="duzeltForm">
        <div class="form-satir">
          <label for="dMarka">Marka</label>
          <input type="text" id="dMarka" value="${kacir(u.marka || '')}" placeholder="Örnek: Lino Perros">
          <p class="yardim">Elle girilen marka, kayıtta &ldquo;elle girildi&rdquo; olarak işaretlenir.</p>
        </div>
        <div class="form-satir">
          <label for="dRaf">Raf</label>
          <input type="text" id="dRaf" value="${kacir(u.raf || '')}" placeholder="Örnek: C-14">
        </div>
        <div class="form-satir">
          <label for="dKod">Ürün kodu / barkod</label>
          <input type="text" id="dKod" value="${kacir(u.urun_kodu || '')}" placeholder="Örnek: 8690000000000">
        </div>
        <button type="submit" class="dugme dugme--birincil">Değişiklikleri kaydet</button>
      </form>
    </div>

    <div class="panel-bolum">
      <h3>Stok hareketi</h3>
      <form id="hareketForm">
        <div class="form-satir">
          <label for="hTip">İşlem</label>
          <select id="hTip">
            <option value="giris">Stok girişi (ekle)</option>
            <option value="cikis">Stok çıkışı (düş)</option>
            <option value="duzeltme">Sayım düzeltmesi (yeni adet)</option>
          </select>
        </div>
        <div class="form-satir">
          <label for="hMiktar">Miktar</label>
          <input type="number" id="hMiktar" min="0" value="1" required>
        </div>
        <div class="form-satir">
          <label for="hAciklama">Açıklama</label>
          <input type="text" id="hAciklama" placeholder="Örnek: sevkiyat, sayım farkı">
          <p class="yardim">Hareket kayıtları silinmez; bu açıklama kalıcı olarak saklanır.</p>
        </div>
        <button type="submit" class="dugme dugme--birincil">Hareketi işle</button>
      </form>
    </div>

    <div class="panel-bolum">
      <h3>Ürünü stoktan çıkar</h3>
      <p class="yardim" style="margin-bottom: var(--a-3)">
        Kayıt ve hareket geçmişi silinmez, ürün yalnızca aramadan kaldırılır.
      </p>
      <button class="dugme dugme--tehlike" id="silDugme">Stoktan çıkar</button>
    </div>`}

    <div class="panel-bolum">
      <h3>Hareket geçmişi</h3>
      ${hareketler
        ? `<ul class="hareket-liste">${hareketler}</ul>`
        : '<p class="yardim">Henüz hareket kaydı yok.</p>'}
    </div>`;
}

async function panelAc(kimlik) {
  durum.secili = kimlik;
  document.querySelectorAll('.urun-kart').forEach((k) => {
    k.setAttribute('aria-current', String(k.dataset.kimlik === kimlik));
  });

  const panel = $('#panel');
  panel.dataset.acik = 'true';
  panel.setAttribute('aria-hidden', 'false');
  $('#panelGovde').innerHTML = '<div class="iskelet"></div><div class="iskelet"></div>';

  try {
    const u = await api(`/api/urun/${encodeURIComponent(kimlik)}`);
    $('#panelBaslik').textContent = urunAdi(u);
    $('#panelGovde').innerHTML = panelHtml(u);
    panelOlaylariBagla(u);
  } catch (hata) {
    $('#panelGovde').innerHTML =
      `<div class="uyari uyari--hata"><span>${kacir(hata.message)}</span></div>`;
  }
}

function panelKapat() {
  const panel = $('#panel');
  panel.dataset.acik = 'false';
  panel.setAttribute('aria-hidden', 'true');
  durum.secili = null;
  document.querySelectorAll('.urun-kart').forEach((k) => k.setAttribute('aria-current', 'false'));
}

function panelOlaylariBagla(u) {
  const duzelt = $('#duzeltForm');
  if (duzelt) {
    duzelt.addEventListener('submit', async (olay) => {
      olay.preventDefault();
      const govde = {
        marka: $('#dMarka').value.trim() || null,
        raf: $('#dRaf').value.trim() || null,
        urun_kodu: $('#dKod').value.trim() || null,
      };
      try {
        await api(`/api/urun/${encodeURIComponent(u.kimlik)}`, {
          method: 'PATCH', body: JSON.stringify(govde),
        });
        bildir('Değişiklikler kaydedildi.', 'olumlu');
        await Promise.all([panelAc(u.kimlik), ozetYukle()]);
        yenile();
      } catch (hata) {
        bildir(hata.message, 'hata');
      }
    });
  }

  const hareket = $('#hareketForm');
  if (hareket) {
    hareket.addEventListener('submit', async (olay) => {
      olay.preventDefault();
      const govde = {
        tip: $('#hTip').value,
        miktar: Number($('#hMiktar').value),
        aciklama: $('#hAciklama').value.trim(),
      };
      try {
        const sonuc = await api(`/api/urun/${encodeURIComponent(u.kimlik)}/hareket`, {
          method: 'POST', body: JSON.stringify(govde),
        });
        bildir(`Stok güncellendi: ${sonuc.onceki} → ${sonuc.sonraki}`, 'olumlu');
        await Promise.all([panelAc(u.kimlik), ozetYukle()]);
        yenile();
      } catch (hata) {
        // İş kuralı ihlali burada görünür oluyor ("stokta 3 varken 5 çıkamaz").
        bildir(hata.message, 'hata');
      }
    });
  }

  const sil = $('#silDugme');
  if (sil) {
    sil.addEventListener('click', () => {
      $('#onayBaslik').textContent = 'Ürünü stoktan çıkar';
      $('#onayMetin').textContent =
        `"${urunAdi(u)}" stoktan çıkarılacak. Kayıt ve hareket geçmişi saklanır, ürün aramada görünmez.`;
      $('#onayGerekce').value = '';
      const pencere = $('#onayPencere');
      pencere.returnValue = '';
      pencere.showModal();

      pencere.addEventListener('close', async function birKez() {
        pencere.removeEventListener('close', birKez);
        if (pencere.returnValue !== 'onay') return;
        try {
          const gerekce = encodeURIComponent($('#onayGerekce').value.trim());
          await api(`/api/urun/${encodeURIComponent(u.kimlik)}?aciklama=${gerekce}`,
            { method: 'DELETE' });
          bildir('Ürün stoktan çıkarıldı.', 'olumlu');
          panelKapat();
          await ozetYukle();
          yenile();
        } catch (hata) {
          bildir(hata.message, 'hata');
        }
      });
    });
  }
}

function yenile() {
  if (durum.gorunum === 'arama') aramaYap(durum.sorgu);
  else listeYukle(durum.gorunum === 'eksik');
}

/* ------------------------------------------------------------------ */
/* Olaylar                                                             */
/* ------------------------------------------------------------------ */

let aramaZamanlayici;

$('#aramaGirdi').addEventListener('input', (olay) => {
  // Her tuşta sunucuya gitmek hem gereksiz hem yavaş; kullanıcı yazmayı
  // bıraktıktan 320 ms sonra aranıyor.
  clearTimeout(aramaZamanlayici);
  const deger = olay.target.value;
  aramaZamanlayici = setTimeout(() => aramaYap(deger), 320);
});

document.querySelectorAll('[data-ornek]').forEach((d) => {
  d.addEventListener('click', () => {
    $('#aramaGirdi').value = d.dataset.ornek;
    // `gorunumDegistir` kutunun içeriğiyle zaten arama yapıyor; ayrıca
    // `aramaYap` çağırmak aynı isteği iki kez gönderiyordu.
    gorunumDegistir('arama');
  });
});

document.querySelectorAll('.yan-dugme').forEach((d) => {
  d.addEventListener('click', () => gorunumDegistir(d.dataset.gorunum));
});

$('#sonucAlani').addEventListener('click', (olay) => {
  const kart = olay.target.closest('.urun-kart');
  if (kart) panelAc(kart.dataset.kimlik);
});

$('#panelKapat').addEventListener('click', panelKapat);

document.addEventListener('keydown', (olay) => {
  if (olay.key === 'Escape' && $('#panel').dataset.acik === 'true') {
    panelKapat();
  }
  // "/" ile aramaya odaklan — bir metin alanında değilsek.
  if (olay.key === '/' && !['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement.tagName)) {
    olay.preventDefault();
    gorunumDegistir('arama');
    $('#aramaGirdi').focus();
  }
});

/* ------------------------------------------------------------------ */
/* Açılış                                                              */
/* ------------------------------------------------------------------ */

ozetYukle();
durumGoster({
  baslik: 'Aramaya başlayın',
  metin: 'Aradığınız ürünü günlük dille yazın. Ürün kodu bilmenize gerek yok. Kısayol: "/" tuşu aramaya odaklanır.',
});


/* ------------------------------------------------------------------ */
/* Ürün ekleme                                                         */
/* ------------------------------------------------------------------ */

/* Yüklenmiş fotoğrafın sunucudaki adı. Form ancak bu doluyken gönderilebilir:
   kayıt fotoğrafa bağlı, kimliği fotoğrafın içeriğinden üretiliyor. */
let yuklenenFoto = null;

function ekleDurumTazele() {
  const hazir = yuklenenFoto !== null && $('#eKategori').value.trim() !== '';
  $('#ekleDugme').disabled = !hazir;
  const uyari = $('#ekleUyari');
  if (yuklenenFoto === null) uyari.textContent = 'Önce bir fotoğraf yükleyin.';
  else if ($('#eKategori').value.trim() === '') uyari.textContent = 'Kategori zorunlu.';
  else uyari.textContent = '';
}

async function kategorileriDoldur() {
  // Var olan kategorilerden öneri listesi: operatör her seferinde yeni bir
  // yazım uydurmasın ("el çantası" / "el cantasi" / "çanta" ayrı kategoriler
  // olurdu ve arama dağılırdı).
  try {
    const veri = await api('/api/urunler?limit=200');
    const kategoriler = [...new Set(
      (veri.urunler || []).map((u) => u.kategori).filter(Boolean)
    )].sort((a, b) => a.localeCompare(b, 'tr'));
    $('#kategoriListesi').innerHTML = kategoriler
      .map((k) => `<option value="${kacir(k)}">`).join('');
  } catch { /* öneri listesi olmasa da form çalışır */ }
}

async function fotoYukle(dosya) {
  if (!dosya) return;

  const govde = new FormData();
  govde.append('dosya', dosya);

  $('#fotoDurum').textContent = 'Yükleniyor…';
  $('#fotoDurum').className = 'foto-durum';
  $('#fotoOnizleme').hidden = false;
  $('#fotoOnizlemeGorsel').src = URL.createObjectURL(dosya);
  $('#fotoBirak').hidden = true;

  try {
    // FormData gönderirken Content-Type ELLE verilmemeli; tarayıcının
    // ürettiği multipart sınırı (boundary) başlıkta yer almalı.
    const yanit = await fetch('/api/foto', { method: 'POST', body: govde });
    if (!yanit.ok) {
      const hata = await yanit.json().catch(() => ({}));
      throw new Error(hata.detail || `Yükleme başarısız (${yanit.status})`);
    }
    const sonuc = await yanit.json();

    if (sonuc.zaten_kayitli) {
      // Kimlik içerikten üretildiği için bu kesin bir tespit, tahmin değil.
      yuklenenFoto = null;
      const ad = sonuc.kayit
        ? [sonuc.kayit.marka, sonuc.kayit.renk, sonuc.kayit.kategori]
            .filter(Boolean).join(' ')
        : 'bir kayıt';
      $('#fotoDurum').textContent =
        `Bu fotoğraf zaten kayıtlı: ${ad}. Aynı ürünü ikinci kez eklemek yerine `
        + 'mevcut kaydın adedini artırın.';
      $('#fotoDurum').className = 'foto-durum foto-durum--uyari';
    } else {
      yuklenenFoto = sonuc.dosya;
      $('#fotoDurum').textContent = 'Fotoğraf hazır.';
      $('#fotoDurum').className = 'foto-durum foto-durum--iyi';
      oznitelikCikar(sonuc.dosya);
    }
  } catch (hata) {
    yuklenenFoto = null;
    $('#fotoDurum').textContent = hata.message;
    $('#fotoDurum').className = 'foto-durum foto-durum--hata';
  }
  ekleDurumTazele();
}

function fotoSifirla() {
  yuklenenFoto = null;
  $('#fotoGirdi').value = '';
  $('#fotoOnizleme').hidden = true;
  $('#fotoBirak').hidden = false;
  // Önceki fotoğrafın çıkarım sonucu ekranda kalmamalı: yeni fotoğraf için
  // "3 alan dolduruldu" yazması yanlış bilgi olurdu.
  $('#vlmDurum').hidden = true;
  $('#vlmAciklama').hidden = false;
  vlmIsaretiTemizle();
  ekleDurumTazele();
}

$('#fotoGirdi').addEventListener('change', (olay) => {
  fotoYukle(olay.target.files[0]);
});

$('#fotoKaldir').addEventListener('click', fotoSifirla);

['dragenter', 'dragover'].forEach((olayAdi) => {
  $('#fotoBirak').addEventListener(olayAdi, (olay) => {
    olay.preventDefault();
    $('#fotoBirak').classList.add('foto-birak--uzerinde');
  });
});

['dragleave', 'drop'].forEach((olayAdi) => {
  $('#fotoBirak').addEventListener(olayAdi, (olay) => {
    olay.preventDefault();
    $('#fotoBirak').classList.remove('foto-birak--uzerinde');
  });
});

$('#fotoBirak').addEventListener('drop', (olay) => {
  fotoYukle(olay.dataTransfer.files[0]);
});

$('#eKategori').addEventListener('input', ekleDurumTazele);

$('#ekleForm').addEventListener('submit', async (olay) => {
  olay.preventDefault();
  if (!yuklenenFoto) return;

  const dugme = $('#ekleDugme');
  dugme.disabled = true;
  dugme.textContent = 'Ekleniyor…';

  try {
    const urun = await api('/api/urun', {
      method: 'POST',
      body: JSON.stringify({
        dosya: yuklenenFoto,
        kategori: $('#eKategori').value.trim(),
        marka: $('#eMarka').value.trim() || null,
        renk: $('#eRenk').value.trim() || null,
        raf: $('#eRaf').value.trim() || null,
        urun_kodu: $('#eKod').value.trim() || null,
        ayirt_edici: $('#eDetay').value.trim() || null,
        adet: Number($('#eAdet').value) || 0,
      }),
    });

    bildir(`Eklendi: ${urunAdi(urun)}`, 'olumlu');
    $('#ekleForm').reset();
    $('#eAdet').value = '1';
    fotoSifirla();
    ozetYukle();
  } catch (hata) {
    bildir(hata.message, 'hata');
  } finally {
    dugme.textContent = 'Stoğa ekle';
    ekleDurumTazele();
  }
});


/* ------------------------------------------------------------------ */
/* Fotoğraftan öznitelik — projenin asıl iddiası                       */
/* ------------------------------------------------------------------ */

/* Operatörün kategori/marka/renk yazması bu sistemin varlık sebebine aykırı:
   o bilgiler fotoğrafta zaten var ve VLM onları okuyabiliyor. Operatöre kalan
   iş DOĞRULAMA — özellikle marka, çünkü ölçüm VLM'in markayı ürünlerin ancak
   %43'ünde okuyabildiğini gösterdi (K11). */

// Yalnızca bu üç alanı VLM dolduruyor. Detay/model alanı BİLEREK burada yok:
// modeli ("iPhone 17") operatör yazar. VLM'in "ayirt_edici" okuması (şekil
// tarifi, örn. "dikdörtgen kamera modülü") o alana konmuyor — operatörün
// gireceği modelle karışırdı.
const VLM_ALANLARI = {
  kategori: '#eKategori',
  marka: '#eMarka',
  renk: '#eRenk',
};

function vlmDurumYaz(metin, tur = '') {
  const kutu = $('#vlmDurum');
  kutu.hidden = false;
  kutu.className = 'vlm-durum' + (tur ? ` vlm-durum--${tur}` : '');
  $('#vlmMetin').textContent = metin;
}

function vlmIsaretiTemizle() {
  Object.values(VLM_ALANLARI).forEach((secici) => {
    const alan = $(secici);
    alan.classList.remove('alan-vlm');
    const rozet = document.querySelector(`label[for="${alan.id}"] .alan-etiket-vlm`);
    if (rozet) rozet.remove();
  });
}

function vlmAlaniIsaretle(secici) {
  const alan = $(secici);
  alan.classList.add('alan-vlm');
  const etiket = document.querySelector(`label[for="${alan.id}"]`);
  if (etiket && !etiket.querySelector('.alan-etiket-vlm')) {
    const rozet = document.createElement('span');
    rozet.className = 'alan-etiket-vlm';
    rozet.textContent = 'fotoğraftan';
    etiket.appendChild(rozet);
  }
  // Operatör alana dokunduğu anda işaret kalkıyor: değer artık onun.
  alan.addEventListener('input', function birKez() {
    alan.classList.remove('alan-vlm');
    const r = etiket && etiket.querySelector('.alan-etiket-vlm');
    if (r) r.remove();
    alan.removeEventListener('input', birKez);
  });
}

function ozniteligiFormaYaz(oznitelik) {
  const bosDegerler = ['', 'bilinmiyor', 'bilinmeyen', 'yok'];
  let yazilan = 0;

  for (const [anahtar, secici] of Object.entries(VLM_ALANLARI)) {
    const deger = (oznitelik[anahtar] || '').toString().trim();
    if (!deger || bosDegerler.includes(deger.toLowerCase())) continue;
    $(secici).value = deger;
    vlmAlaniIsaretle(secici);
    yazilan += 1;
  }
  return yazilan;
}

async function oznitelikCikar(dosya) {
  vlmIsaretiTemizle();
  $('#vlmAciklama').hidden = true;

  let baslangic;
  try {
    baslangic = await api(
      `/api/is/oznitelik?dosya=${encodeURIComponent(dosya)}`, { method: 'POST' }
    );
  } catch (hata) {
    vlmDurumYaz(`Öznitelik çıkarımı başlatılamadı: ${hata.message}`, 'hata');
    return;
  }

  // Bekleme süresi modelin yüklü olup olmamasına göre kat kat değişiyor;
  // hangi durumda olduğumuz söylenmeli, yoksa kullanıcı donmuş sanıyor.
  vlmDurumYaz(
    baslangic.model_hazir
      ? 'Fotoğraf okunuyor… (birkaç saniye)'
      : 'Model ilk kez yükleniyor… (bir buçuk dakika kadar sürebilir; '
        + 'sonraki ürünler saniyeler içinde gelecek)'
  );

  const bitis = Date.now() + 6 * 60 * 1000;
  while (Date.now() < bitis) {
    await new Promise((c) => setTimeout(c, 1500));

    let is;
    try {
      is = await api(`/api/is/${baslangic.is_id}`);
    } catch (hata) {
      vlmDurumYaz(`Durum alınamadı: ${hata.message}`, 'hata');
      return;
    }

    if (is.durum === 'bitti') {
      const yazilan = ozniteligiFormaYaz(is.oznitelik || {});
      if (yazilan === 0) {
        vlmDurumYaz(
          'Fotoğraftan okunabilir bilgi çıkmadı. Alanları elle doldurun.', 'hata'
        );
      } else {
        vlmDurumYaz(
          `${yazilan} alan fotoğraftan dolduruldu. Lütfen doğrulayın; `
          + 'gerekirse modeli/detayı elle ekleyin.',
          'bitti'
        );
      }
      ekleDurumTazele();
      return;
    }

    if (is.durum === 'hata') {
      vlmDurumYaz(
        `Fotoğraf okunamadı: ${is.hata || 'bilinmeyen hata'}. `
        + 'Alanları elle doldurabilirsiniz.', 'hata'
      );
      return;
    }

    if (is.durum === 'model_yukleniyor') {
      vlmDurumYaz('Model yükleniyor… (yalnızca ilk üründe)');
    } else if (is.durum === 'cikariliyor') {
      vlmDurumYaz('Fotoğraf okunuyor…');
    }
  }

  vlmDurumYaz(
    'Çıkarım beklenenden uzun sürdü. Alanları elle doldurabilirsiniz.', 'hata'
  );
}
