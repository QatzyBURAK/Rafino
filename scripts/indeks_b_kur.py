"""İndeks B'yi kurar: VLM özniteliklerinden Türkçe açıklama üretip gömer.

Girdi: data/oznitelikler.jsonl  (scripts/vlm_toplu.py üretir)
Çıktı: ChromaDB'deki metin koleksiyonu

Kimlikler İndeks A ile AYNI (fotoğrafın içerik özeti), böylece RRF birleştirmesi
sırasında iki indeksin sonuçları doğrudan eşleşiyor.

    .venv\\Scripts\\python.exe scripts\\indeks_b_kur.py [piksel|vlm]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.db import chroma  # noqa: E402
from src.ingest.aciklama import aranabilir_metin  # noqa: E402
from src.ingest.renk import renk_bul  # noqa: E402
from src.models.embedder import MetinEmbedder  # noqa: E402


def main() -> int:
    renk_kaynagi = sys.argv[1] if len(sys.argv) > 1 else "piksel"
    if renk_kaynagi not in {"piksel", "vlm"}:
        print("[!] Renk kaynağı 'piksel' veya 'vlm' olmalı")
        return 1

    yol = config.VERI_DIZINI / "oznitelikler.jsonl"
    if not yol.exists():
        print(f"[!] Öznitelik dosyası yok: {yol}")
        print("    Önce: .venv\\Scripts\\python.exe scripts\\vlm_toplu.py")
        return 1

    kayitlar = [
        json.loads(s) for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()
    ]
    print(f"[i] {len(kayitlar)} öznitelik kaydı, renk kaynağı: {renk_kaynagi}")

    kimlikler: list[str] = []
    metinler: list[str] = []
    metalar: list[dict] = []

    for kayit in kayitlar:
        foto = config.FOTO_DIZINI / kayit["dosya"]
        if not foto.exists():
            print(f"  [!] Fotoğraf yok, atlanıyor: {kayit['dosya']}")
            continue

        if renk_kaynagi == "piksel":
            renk, pay = renk_bul(foto)
        else:
            renk, pay = str(kayit.get("renk", "")).strip().lower(), 1.0

        metin = aranabilir_metin(kayit, renk)
        kimlikler.append(chroma.urun_kimligi(foto))
        metinler.append(metin)
        metalar.append({
            "dosya": kayit["dosya"],
            "aciklama": metin,
            "kategori": str(kayit.get("kategori", "")),
            "marka": str(kayit.get("marka", "")),
            "renk": renk,
            "renk_kaynagi": renk_kaynagi,
            "renk_payi": round(float(pay), 3),
        })

    print(f"\n=== Üretilen açıklamalardan örnekler ===")
    for m in metinler[:8]:
        print(f"  {m}")

    print(f"\n[i] {len(metinler)} açıklama gömülüyor...")
    embedder = MetinEmbedder()
    vektorler = embedder.belgeleri_gom(metinler, ilerleme=True)

    koleksiyon = chroma.metin_koleksiyon()
    koleksiyon.upsert(
        ids=kimlikler,
        embeddings=[v.tolist() for v in vektorler],
        metadatas=metalar,
    )
    print(f"[+] Bitti. İndeks B'de {koleksiyon.count()} kayıt var.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
