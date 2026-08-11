"""Model ağırlıklarını önceden indirir.

Ağırlıkları yalnızca diske çeker, GPU'ya yüklemez — bu yüzden indirme sırasında
VRAM harcanmaz ve başka iş yapılabilir.

    .venv\\Scripts\\python.exe scripts\\indir_modeller.py
"""

from huggingface_hub import snapshot_download

INDIRILECEK = [
    ("Qwen/Qwen3-VL-Embedding-2B", "görsel embedding (İndeks A)"),
    ("intfloat/multilingual-e5-large", "metin embedding (İndeks B)"),
]


def main() -> int:
    for model_id, rol in INDIRILECEK:
        print(f"\n=== {model_id} — {rol} ===")
        try:
            yol = snapshot_download(model_id)
            print(f"[ OK ] {yol}")
        except Exception as exc:  # noqa: BLE001
            print(f"[HATA] {type(exc).__name__}: {exc}")
            return 1
    print("\nTüm ağırlıklar hazır.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
