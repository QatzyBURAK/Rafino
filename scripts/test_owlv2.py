"""OWLv2 izole testi — metin promptlu nesne tespiti.

SAM 3'ün yerine değerlendiriliyor. SAM maske döndürüyor, burada gereken kutu;
OWLv2 metin promptu alıp doğrudan kutu döndürüyor ve beşte biri kadar büyük.

Ölçtüğü şey: bir fotoğrafta kaç ürün olduğunu güvenilir şekilde bulabiliyor
muyuz? Bulabiliyorsak çoklu ürün fotoğrafları parçalanıp tek tek kaydedilir;
bulamıyorsak "tek fotoğraf = tek ürün" kuralı konur ve ihlali AÇIKÇA bildirilir
(sessiz yedek plan, çalışmayan modeli çalışıyor sanmaya yol açar).

Kullanım:
    .venv\\Scripts\\python.exe scripts\\test_owlv2.py <foto> [esik]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Depoda karşılaşılan ürün türleri. OWLv2 açık sözlüklü: bu listede olmayan
# bir şey de "product" sorgusuyla yakalanabiliyor.
SORGULAR = [
    "a product", "a bag", "a handbag", "a shoe", "a watch",
    "a wallet", "sunglasses", "a box", "a piece of clothing", "a bottle",
]


def vram_gb() -> tuple[float, float]:
    import torch

    if not torch.cuda.is_available():
        return (0.0, 0.0)
    bos, toplam = torch.cuda.mem_get_info()
    return ((toplam - bos) / 1024**3, toplam / 1024**3)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    yol = Path(sys.argv[1])
    esik = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    if not yol.exists():
        print(f"[!] Dosya yok: {yol}")
        return 1

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Owlv2ForObjectDetection

    from src import config

    once, toplam = vram_gb()
    print(f"[i] Yükleme öncesi VRAM: {once:.2f} / {toplam:.1f} GB")

    model_id = "google/owlv2-base-patch16-ensemble"
    print(f"[i] Yükleniyor: {model_id}")
    t0 = time.perf_counter()
    islemci = AutoProcessor.from_pretrained(model_id)
    model = Owlv2ForObjectDetection.from_pretrained(model_id).to("cuda").eval()
    print(f"[+] Yüklendi. {time.perf_counter() - t0:.1f} sn")

    sonra, _ = vram_gb()
    print(f"[+] VRAM: {sonra:.2f} GB (model ~{sonra - once:.2f} GB)")

    gorsel = Image.open(yol).convert("RGB")
    print(f"[i] Fotoğraf: {yol.name}  {gorsel.size[0]}x{gorsel.size[1]}")

    girdi = islemci(text=[SORGULAR], images=gorsel, return_tensors="pt").to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        cikti = model(**girdi)
    print(f"[i] Çıkarım: {time.perf_counter() - t0:.2f} sn")

    sonuc = islemci.post_process_grounded_object_detection(
        outputs=cikti,
        target_sizes=torch.tensor([gorsel.size[::-1]]).to("cuda"),
        threshold=esik,
    )[0]

    skorlar = sonuc["scores"].tolist()
    etiketler = sonuc["labels"].tolist()
    kutular = sonuc["boxes"].tolist()

    print(f"\n[i] Eşik {esik} üzerinde {len(skorlar)} kutu bulundu\n")
    if not skorlar:
        print("[!] Hiç kutu yok. Eşiği düşürmeyi dene.")
        return 1

    sirali = sorted(zip(skorlar, etiketler, kutular), key=lambda x: -x[0])
    print(f"{'#':<4}{'skor':<8}{'etiket':<22}kutu (x1,y1,x2,y2)")
    print("-" * 72)
    for i, (skor, etiket, kutu) in enumerate(sirali[:25], 1):
        k = ", ".join(f"{v:.0f}" for v in kutu)
        print(f"{i:<4}{skor:<8.3f}{SORGULAR[etiket]:<22}({k})")

    zirve, _ = vram_gb()
    print(f"\n[i] Zirve VRAM: {zirve:.2f} / {toplam:.1f} GB")
    print(f"[i] Toplam tespit: {len(skorlar)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
