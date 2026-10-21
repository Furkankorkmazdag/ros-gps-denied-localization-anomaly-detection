import os
import torch

from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine


# ============================================================
# PROJE DİZİNİ
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATASET
# ============================================================

DATASET_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "train"
)

NORMAL_DIR = os.path.join(
    DATASET_DIR,
    "normal"
)


# ============================================================
# SONUÇ DİZİNİ
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "patchcore"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# BAŞLANGIÇ
# ============================================================

print("=" * 70)
print("PATCHCORE EĞİTİMİ")
print("=" * 70)


# ============================================================
# PYTORCH / CUDA
# ============================================================

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA kullanılabilir:", torch.cuda.is_available())


if not torch.cuda.is_available():

    print()
    print("[HATA] CUDA kullanılamıyor!")
    print("[HATA] PatchCore eğitimi durduruldu.")

    raise SystemExit(1)


print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print()


# ============================================================
# DATASET KONTROLÜ
# ============================================================

if not os.path.isdir(NORMAL_DIR):

    print("[HATA] Normal dataset bulunamadı:")
    print(NORMAL_DIR)

    raise SystemExit(1)


image_files = [
    file_name
    for file_name in os.listdir(NORMAL_DIR)
    if file_name.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    )
]


print(
    "Normal görüntü sayısı:",
    len(image_files)
)


if len(image_files) == 0:

    print("[HATA] Normal görüntü bulunamadı!")

    raise SystemExit(1)


# ============================================================
# FOLDER DATAMODULE
# ============================================================

print()
print("Folder datamodule oluşturuluyor...")


datamodule = Folder(

    name="tunnel_normal",

    root=DATASET_DIR,

    normal_dir="normal",

    # Henüz anomaly görüntülerimiz yok.
    abnormal_dir=None,

    # Ayrı normal test klasörü de henüz yok.
    normal_test_dir=None,

    # Normal görüntülerin tamamını train'e ver.
    normal_split_ratio=0.0,

    # Batch
    train_batch_size=8,
    eval_batch_size=8,

    # RTX 3050 için düşük worker sayısı
    num_workers=2,

    # Henüz test görüntüsü olmadığı için
    # test'i klasörden bekleme.
    test_split_mode="from_dir",

    # Validation için şimdilik split yapmıyoruz.
    val_split_mode="same_as_test",

    val_split_ratio=0.0,

    seed=42,
)


# ============================================================
# PATCHCORE MODELİ
# ============================================================

print()
print("PatchCore modeli oluşturuluyor...")


model = Patchcore(

    backbone="wide_resnet50_2",

    layers=[
        "layer2",
        "layer3",
    ],

    pre_trained=True,

    # Memory bank'in %10'u
    coreset_sampling_ratio=0.1,

    # Nearest-neighbor sayısı
    num_neighbors=9,

)


# ============================================================
# ENGINE
# ============================================================

print()
print("Anomalib Engine oluşturuluyor...")


engine = Engine(

    default_root_dir=RESULTS_DIR,

    accelerator="gpu",

    devices=1,

)


# ============================================================
# PATCHCORE FIT
# ============================================================

print()
print("=" * 70)
print("NORMAL GÖRÜNTÜLERDEN MEMORY BANK OLUŞTURULUYOR")
print("=" * 70)
print()

engine.fit(

    model=model,

    datamodule=datamodule,

)


# ============================================================
# TAMAMLANDI
# ============================================================

print()
print("=" * 70)
print("PATCHCORE TAMAMLANDI")
print("=" * 70)

print()
print("Sonuç klasörü:")
print(RESULTS_DIR)

print()
print("En iyi model:")
print(engine.best_model_path)

print()
print("GPU belleği:")

allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)

print(
    f"Allocated: {allocated:.1f} MB"
)

print(
    f"Reserved : {reserved:.1f} MB"
)
