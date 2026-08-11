"""Faz 0 — görsel indeks (İndeks A) üzerinde terminal araması.

Bu aşamada yalnızca fotoğraf vektörleri var. VLM açıklamalarının metin indeksi
(İndeks B) ve anahtar kelime indeksi (İndeks C) sonraki fazlarda ekleniyor.

Faz 0'ın amacı ölçüm için taban çizgisi üretmek: "mavi valiz" tipi sorgular tek
görsel indeksle ne kadar iyi çalışıyor? Bu sayı olmadan hibrit yapının kazancı
iddiadan ibaret kalır.

    .venv\\Scripts\\python.exe scripts\\faz0.py ekle data\\photos
    .venv\\Scripts\\python.exe scripts\\faz0.py ara "mavi valiz"
    .venv\\Scripts\\python.exe scripts\\faz0.py benzer data\\photos\\foo.jpg
    .venv\\Scripts\\python.exe scripts\\faz0.py durum
    .venv\\Scripts\\python.exe scripts\\faz0.py sifirla
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma  # noqa: E402
from src.models.embedder import GorselEmbedder  # noqa: E402


def _fotograflari_bul(klasor: Path) -> list[Path]:
    return sorted(
        p for p in klasor.rglob("*") if p.suffix.lower() in config.RESIM_UZANTILARI
    )


def cmd_ekle(klasor_str: str) -> int:
    klasor = Path(klasor_str) if klasor_str else config.FOTO_DIZINI
    if not klasor.exists():
        print(f"[!] Klasör yok: {klasor}")
        return 1

    yollar = _fotograflari_bul(klasor)
    if not yollar:
        print(f"[!] {klasor} içinde resim yok.")
        return 1

    # İçerik özetinden kimlik: aynı dosyanın kopyaları tek kayda düşer.
    # Ayıklama gömmeden ÖNCE yapılır, yoksa kopyalar boşuna GPU'da işlenir.
    benzersiz: dict[str, Path] = {}
    kopyalar: list[tuple[Path, Path]] = []
    for p in yollar:
        kimlik = chroma.urun_kimligi(p)
        if kimlik in benzersiz:
            kopyalar.append((p, benzersiz[kimlik]))
        else:
            benzersiz[kimlik] = p

    if kopyalar:
        print(f"[i] {len(kopyalar)} dosya birebir kopya, atlanıyor:")
        for yeni, ilk in kopyalar:
            print(f"      {yeni.name}  ==  {ilk.name}")

    kimlikler = list(benzersiz.keys())
    islenecek = list(benzersiz.values())

    print(f"[i] {len(islenecek)} benzersiz fotoğraf gömülüyor...")
    embedder = GorselEmbedder()
    vektorler = embedder.gorselleri_gom(islenecek)

    simdi = datetime.now(timezone.utc).isoformat()
    koleksiyon = chroma.gorsel_koleksiyon()
    koleksiyon.upsert(
        ids=kimlikler,
        embeddings=[v.tolist() for v in vektorler],
        metadatas=[
            {
                "dosya": p.name,
                "yol": str(p.resolve()),
                "boyut_bayt": p.stat().st_size,
                "eklendi": simdi,
            }
            for p in islenecek
        ],
    )
    print(f"[+] Bitti. Görsel indekste toplam {koleksiyon.count()} ürün var.")
    return 0


def _sonuclari_yazdir(sonuc: dict) -> None:
    kimlikler = sonuc["ids"][0]
    if not kimlikler:
        print("[!] Sonuç yok. Önce 'ekle' çalıştır.")
        return

    metalar = sonuc["metadatas"][0]
    mesafeler = sonuc["distances"][0]
    print(f"\n{'#':<3} {'dosya':<34} {'benzerlik':>9}")
    print("-" * 50)
    for sira, (meta, mesafe) in enumerate(zip(metalar, mesafeler), 1):
        # ChromaDB cosine "distance" döndürür; benzerlik = 1 - mesafe.
        benzerlik = 1 - mesafe
        print(f"{sira:<3} {meta['dosya']:<34} {benzerlik:>9.3f}")
    print()


def cmd_ara(sorgu: str) -> int:
    if not sorgu:
        print("[!] Sorgu boş. Örnek: ara \"mavi valiz\"")
        return 1
    print(f"[i] Sorgu: '{sorgu}'")
    embedder = GorselEmbedder()
    vektor = embedder.sorguyu_gom([sorgu])[0]
    sonuc = chroma.gorsel_koleksiyon().query(
        query_embeddings=[vektor.tolist()], n_results=config.VARSAYILAN_SONUC
    )
    _sonuclari_yazdir(sonuc)
    return 0


def cmd_benzer(foto_str: str) -> int:
    foto = Path(foto_str)
    if not foto.exists():
        print(f"[!] Dosya yok: {foto}")
        return 1
    print(f"[i] Görsel sorgu: {foto.name}")
    embedder = GorselEmbedder()
    vektor = embedder.gorselleri_gom([foto], ilerleme=False)[0]
    sonuc = chroma.gorsel_koleksiyon().query(
        query_embeddings=[vektor.tolist()], n_results=config.VARSAYILAN_SONUC
    )
    _sonuclari_yazdir(sonuc)
    return 0


def cmd_durum(_: str = "") -> int:
    gorsel = chroma.gorsel_koleksiyon()
    metin = chroma.metin_koleksiyon()
    print(f"[i] Veritabanı : {config.CHROMA_DIZINI}")
    print(f"[i] Fotoğraflar: {config.FOTO_DIZINI}")
    print(f"[i] İndeks A (görsel): {gorsel.count()} kayıt")
    print(f"[i] İndeks B (metin) : {metin.count()} kayıt   <- Faz 1'de dolacak")
    diskteki = len(_fotograflari_bul(config.FOTO_DIZINI)) if config.FOTO_DIZINI.exists() else 0
    print(f"[i] Klasördeki fotoğraf: {diskteki}")
    if diskteki > gorsel.count():
        print("[!] Diskte indekslenmemiş fotoğraf var — 'ekle' çalıştır.")
    return 0


def cmd_sifirla(_: str = "") -> int:
    onay = input("Her iki indeks de silinecek. Emin misin? (evet/hayir): ").strip().lower()
    if onay != "evet":
        print("[i] Vazgeçildi.")
        return 0
    chroma.sifirla()
    print("[+] İndeksler silindi. Fotoğraflar diskte duruyor.")
    return 0


KOMUTLAR = {
    "ekle": cmd_ekle,
    "ara": cmd_ara,
    "benzer": cmd_benzer,
    "durum": cmd_durum,
    "sifirla": cmd_sifirla,
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    komut = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else ""
    islev = KOMUTLAR.get(komut)
    if islev is None:
        print(f"[!] Bilinmeyen komut: {komut}")
        print(f"    Komutlar: {' | '.join(KOMUTLAR)}")
        return 1
    return islev(arg)


if __name__ == "__main__":
    raise SystemExit(main())
