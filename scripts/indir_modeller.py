"""Model ağırlıklarını önceden indirir.

Ağırlıkları yalnızca diske çeker, GPU'ya yüklemez — bu yüzden indirme sırasında
VRAM harcanmaz ve başka iş yapılabilir. Yarıda kalırsa tekrar çalıştır, kaldığı
yerden devam eder.

    .venv\\Scripts\\python.exe scripts\\indir_modeller.py
"""

import os

# Xet depolama arka ucu, kimliksiz isteklerde uzun indirmelerde kopuyor
# ("CAS Client Error"). Klasik HTTP indirmesi daha yavaş ama kararlı.
# `hf auth login` yapıldıktan sonra bu satır kaldırılabilir.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Windows'ta sembolik bağlantı desteği kapalı; uyarıyı susturuyoruz.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download  # noqa: E402

# Kullanmadığımız çerçevelerin ağırlıkları. multilingual-e5-large deposunda
# ONNX ve OpenVINO sürümleri de var; sentence-transformers safetensors okuyor.
# Bunları atlamak indirmeyi belirgin şekilde kısaltıyor.
ATLANACAK = [
    "onnx/**",
    "openvino/**",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "tf_model*",
    "flax_model*",
    "rust_model*",
]

INDIRILECEK = [
    ("Qwen/Qwen3-VL-Embedding-2B", "görsel embedding (İndeks A)"),
    ("intfloat/multilingual-e5-large", "metin embedding (İndeks B)"),
    ("Qwen/Qwen3-VL-4B-Instruct", "öznitelik çıkarımı (VLM)"),
    ("google/owlv2-base-patch16-ensemble", "çoklu ürün tespiti"),
]


def main() -> int:
    for model_id, rol in INDIRILECEK:
        print(f"\n=== {model_id} — {rol} ===", flush=True)
        try:
            yol = snapshot_download(model_id, ignore_patterns=ATLANACAK)
            print(f"[ OK ] {yol}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[HATA] {type(exc).__name__}: {exc}", flush=True)
            print("       Tekrar çalıştır — indirme kaldığı yerden devam eder.", flush=True)
            return 1
    print("\nTüm ağırlıklar hazır.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
