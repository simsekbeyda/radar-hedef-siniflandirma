"""Model tanımları ve kaydetme/yükleme yardımcıları."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import FEATURES, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE


def _preprocessor(scale: bool = False) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", SimpleImputer(strategy="most_frequent"), ["pervane_imzasi"]),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, Pipeline]:
    """Karşılaştırılacak aday modeller (hepsi eksik değer doldurmayı içerir)."""
    return {
        "Karar Ağacı (depth=4)": Pipeline(
            [
                ("prep", _preprocessor()),
                ("clf", DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE)),
            ]
        ),
        "Lojistik Regresyon": Pipeline(
            [
                ("prep", _preprocessor(scale=True)),
                ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("prep", _preprocessor()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=150, max_depth=2, learning_rate=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def save_model(model: Pipeline, path: str | Path = MODEL_PATH, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "meta": meta or {}}, path)
    return path


def load_model(path: str | Path = MODEL_PATH) -> tuple[Pipeline, dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\nÖnce eğitimi çalıştırın: python -m radar_classifier.train"
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle.get("meta", {})


def to_frame(hiz, irtifa, rcs, manevra, pervane) -> pd.DataFrame:
    """Tek bir ölçümü modelin beklediği DataFrame biçimine çevirir."""
    return pd.DataFrame([[hiz, irtifa, rcs, manevra, int(pervane)]], columns=FEATURES)
