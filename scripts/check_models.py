"""Model erişim doğrulaması — ağırlık indirmeden.

Sadece config.json çeker (birkaç KB). Şunları erken yakalar:
  - yanlış yazılmış model kimliği
  - kapılı (gated) model için eksik lisans onayı / token
  - transformers sürümünün mimariyi tanımaması
  - ağ / proxy sorunu

Saatlik indirmeye başlamadan önce çalıştır:

    .venv\\Scripts\\python.exe scripts\\check_models.py
"""

from transformers import AutoConfig

MODELLER = {
    "Qwen/Qwen3-VL-Embedding-2B": "görsel embedding (İndeks A)",
    "Qwen/Qwen3-VL-4B-Instruct": "öznitelik çıkarımı (VLM)",
    "intfloat/multilingual-e5-large": "metin embedding (İndeks B)",
    "google/owlv2-base-patch16-ensemble": "çoklu ürün tespiti (opsiyonel)",
}


def main() -> int:
    basarisiz: list[str] = []

    for model_id, rol in MODELLER.items():
        print(f"\n{model_id}")
        print(f"  rol: {rol}")
        try:
            config = AutoConfig.from_pretrained(model_id)
        except Exception as exc:  # noqa: BLE001 - hangi hata olursa olsun raporla
            print(f"  [HATA] {type(exc).__name__}: {exc}")
            basarisiz.append(model_id)
            continue

        mimari = getattr(config, "model_type", "bilinmiyor")
        print(f"  [ OK ] erişilebilir — mimari: {mimari}")

    print("\n" + "=" * 60)
    if basarisiz:
        print(f"{len(basarisiz)} model erişilemedi:")
        for m in basarisiz:
            print(f"  - {m}")
        print("\nKapılı model ise: model sayfasından lisansı onayla, sonra")
        print("  hf auth login")
        return 1
    print("Tüm modeller erişilebilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
