"""Ortam doğrulama betiği.

Kurulumdan sonra ve her şüphede çalıştır:

    .venv\\Scripts\\python.exe scripts\\check_env.py

Sessizce yanlış çalışan bir ortam, hatalı çalışan bir ortamdan pahalıdır.
Bu betiğin yakaladığı tuzaklar teknik referans dokümanındaki
"acı çekerek öğrenilenler" bölümünden geliyor.
"""

import importlib
import sys

OK = "[ OK ]"
UYARI = "[UYARI]"
HATA = "[HATA]"

sorunlar: list[str] = []


def baslik(metin: str) -> None:
    print(f"\n--- {metin} ---")


def kontrol_python() -> None:
    baslik("Python")
    s = sys.version_info
    print(f"{OK} sürüm {s.major}.{s.minor}.{s.micro}")
    print(f"     {sys.executable}")
    if (s.major, s.minor) != (3, 11):
        mesaj = f"Python 3.11 bekleniyordu, {s.major}.{s.minor} çalışıyor"
        print(f"{UYARI} {mesaj}")
        sorunlar.append(mesaj)
    if ".venv" not in sys.executable:
        mesaj = "Sanal ortam dışında çalışıyorsun — .venv\\Scripts\\python.exe kullan"
        print(f"{UYARI} {mesaj}")
        sorunlar.append(mesaj)


def kontrol_torch() -> None:
    baslik("PyTorch / CUDA")
    try:
        import torch
    except ImportError:
        print(f"{HATA} torch kurulu değil")
        sorunlar.append("torch kurulu değil")
        return

    print(f"{OK} torch {torch.__version__}")

    # Doküman tuzağı: sürüm sonunda +cpu varsa GPU hiç kullanılmıyor demektir.
    if "+cpu" in torch.__version__:
        mesaj = "torch CPU sürümü kurulmuş — cu128 tekerleği ile yeniden kur"
        print(f"{HATA} {mesaj}")
        sorunlar.append(mesaj)
        return

    if not torch.cuda.is_available():
        mesaj = "torch.cuda.is_available() False — sürücü veya kurulum sorunu"
        print(f"{HATA} {mesaj}")
        sorunlar.append(mesaj)
        return

    print(f"{OK} CUDA {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        ozellik = torch.cuda.get_device_properties(i)
        gb = ozellik.total_memory / 1024**3
        print(f"{OK} GPU {i}: {ozellik.name} — {gb:.1f} GB VRAM")
        if gb < 7.5:
            mesaj = f"GPU {i} beklenenden az VRAM bildiriyor ({gb:.1f} GB)"
            print(f"{UYARI} {mesaj}")
            sorunlar.append(mesaj)

    # Gerçek bir tensör işlemi: is_available() True olup çalışmayan kurulumlar var.
    try:
        x = torch.randn(64, 64, device="cuda")
        _ = (x @ x).sum().item()
        torch.cuda.synchronize()
        print(f"{OK} GPU üzerinde matris çarpımı çalıştı")
    except Exception as exc:  # noqa: BLE001 - hangi hata olursa olsun raporla
        mesaj = f"GPU işlemi başarısız: {exc}"
        print(f"{HATA} {mesaj}")
        sorunlar.append(mesaj)


def kontrol_paketler() -> None:
    baslik("Paketler")
    beklenen = [
        "transformers",
        "sentence_transformers",
        "chromadb",
        "fastapi",
        "uvicorn",
        "PIL",
        "cv2",
        "numpy",
        "sklearn",
    ]
    for ad in beklenen:
        try:
            modul = importlib.import_module(ad)
        except ImportError:
            print(f"{HATA} {ad} içe aktarılamadı")
            sorunlar.append(f"{ad} kurulu değil")
            continue
        surum = getattr(modul, "__version__", "sürüm bilinmiyor")
        print(f"{OK} {ad} {surum}")


def kontrol_vram_bosta() -> None:
    baslik("VRAM durumu")
    try:
        import torch

        if not torch.cuda.is_available():
            print("     atlandı (CUDA yok)")
            return
        bos, toplam = torch.cuda.mem_get_info()
        print(f"{OK} boşta {bos / 1024**3:.1f} GB / {toplam / 1024**3:.1f} GB")
        if bos / 1024**3 < 6.0:
            mesaj = "Boştaki VRAM 6 GB'ın altında — başka bir süreç GPU kullanıyor olabilir"
            print(f"{UYARI} {mesaj}")
            sorunlar.append(mesaj)
    except Exception as exc:  # noqa: BLE001
        print(f"{UYARI} VRAM okunamadı: {exc}")


def main() -> int:
    print("=" * 60)
    print("Görsel Tabanlı Akıllı Stok Takip Sistemi — ortam kontrolü")
    print("=" * 60)

    kontrol_python()
    kontrol_torch()
    kontrol_paketler()
    kontrol_vram_bosta()

    print("\n" + "=" * 60)
    if sorunlar:
        print(f"{len(sorunlar)} sorun bulundu:")
        for s in sorunlar:
            print(f"  - {s}")
        return 1
    print("Ortam temiz. Çalışmaya hazır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
