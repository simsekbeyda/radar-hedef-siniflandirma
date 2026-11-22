"""Veri yükleme ve temizleme.

Not: Eksik değerlerin medyanla doldurulması burada YAPILMAZ. Doldurma, modelin
sklearn Pipeline'ı içinde (SimpleImputer) yapılır; böylece medyan yalnızca
eğitim verisinden hesaplanır ve test verisinden bilgi sızmaz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASSES, CLEAN_DATA_PATH, NUMERIC_FEATURES, RAW_DATA_PATH, TARGET


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Ham CSV dosyasını okur."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ham radar verisini temizler.

    1. Tekrar eden satırları siler.
    2. Etiket yazımlarını düzeltir ("KUS", " mini_iha" -> "kus", "mini_iha").
    3. Etiketi olmayan veya bilinmeyen etiketli satırları siler.
    4. Fiziksel olarak imkânsız değerleri (negatif hız/irtifa = sensör hata
       kodu) eksik değere (NaN) çevirir.
    """
    df = df.copy()
    df = df.drop_duplicates()

    df[TARGET] = df[TARGET].astype("string").str.strip().str.lower()
    df = df[df[TARGET].isin(CLASSES)]

    for col in ("hiz_kmh", "irtifa_m", "rcs_m2"):
        df.loc[df[col] < 0, col] = np.nan

    df[TARGET] = df[TARGET].astype(str)
    return df.reset_index(drop=True)


def data_quality_report(df: pd.DataFrame) -> dict:
    """Ham verideki sorunları özetler."""
    return {
        "satir": len(df),
        "eksik_deger": {k: int(v) for k, v in df.isnull().sum().items()},
        "tekrar_eden_satir": int(df.duplicated().sum()),
        "negatif_hiz": int((df["hiz_kmh"] < 0).sum()),
        "negatif_irtifa": int((df["irtifa_m"] < 0).sum()),
        "etiketler": {
            str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()
        },
    }


def save_clean(df: pd.DataFrame, path: str | Path = CLEAN_DATA_PATH) -> Path:
    """Temiz veriyi kaydeder (EDA için; sayısal eksikler medyanla doldurulur)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in NUMERIC_FEATURES:
        out[col] = out[col].fillna(out[col].median())
    out.to_csv(path, index=False)
    return path
