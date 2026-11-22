import numpy as np
import pandas as pd

from radar_classifier.config import CLASSES, TARGET
from radar_classifier.data import clean, load_raw


def _ornek_df():
    return pd.DataFrame(
        {
            "hiz_kmh": [35.0, 35.0, -1.0, 50.0, 60.0],
            "irtifa_m": [387.0, 387.0, 500.0, -999.0, 100.0],
            "rcs_m2": [0.02, 0.02, 0.01, 0.03, 0.05],
            "manevra_skoru": [7.9, 7.9, 3.0, np.nan, 6.0],
            "pervane_imzasi": [1, 1, 0, 1, 0],
            TARGET: ["mini_iha", "mini_iha", " KUS", "Mini_IHA", None],
        }
    )


def test_tekrarlar_silinir_ve_etiketsiz_satir_atilir():
    df = clean(_ornek_df())
    assert len(df) == 3


def test_etiketler_normalize_edilir():
    df = clean(_ornek_df())
    assert set(df[TARGET]) <= set(CLASSES)
    assert "kus" in set(df[TARGET])


def test_negatif_degerler_nan_olur():
    df = clean(_ornek_df())
    assert (df["hiz_kmh"].dropna() >= 0).all()
    assert (df["irtifa_m"].dropna() >= 0).all()
    assert df["hiz_kmh"].isna().sum() == 1
    assert df["irtifa_m"].isna().sum() == 1


def test_gercek_veri_temizlenir():
    df = clean(load_raw())
    assert len(df) == 198
    assert sorted(df[TARGET].unique()) == sorted(CLASSES)
    assert not df.duplicated().any()
