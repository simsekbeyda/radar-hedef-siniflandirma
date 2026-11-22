"""Proje genelinde kullanılan sabitler ve dosya yolları."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "radar_hedefleri.csv"
CLEAN_DATA_PATH = DATA_DIR / "processed" / "radar_hedefleri_temiz.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "radar_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

FEATURES = ["hiz_kmh", "irtifa_m", "rcs_m2", "manevra_skoru", "pervane_imzasi"]
NUMERIC_FEATURES = ["hiz_kmh", "irtifa_m", "rcs_m2", "manevra_skoru"]
TARGET = "hedef_turu"

CLASSES = ["kus", "mini_iha", "sabit_kanat_iha", "helikopter_ucak"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Operatöre gösterilecek okunur ad ve önerilen işlem
ACIKLAMALAR = {
    "kus": ("KUŞ", "Tehdit değil. Alarm gerekmiyor, izlemeye devam edilebilir."),
    "mini_iha": (
        "MİNİ İHA (drone)",
        "DİKKAT! İzinsiz drone olabilir. Operatörü uyar ve hedefi yakından izle.",
    ),
    "sabit_kanat_iha": (
        "SABİT KANATLI İHA",
        "Kimlik doğrulaması yap: Bu bölgede uçuş izni olan bir İHA mı?",
    ),
    "helikopter_ucak": (
        "HELİKOPTER / UÇAK",
        "İnsanlı hava aracı. Uçuş planı ve transponder bilgisiyle karşılaştır.",
    ),
}

# Bu eşiğin altındaki güvenle yapılan tahminler operatör kontrolüne düşer
GUVEN_ESIGI = 80.0
