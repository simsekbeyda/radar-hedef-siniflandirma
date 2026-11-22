"""Modelleri karşılaştırır, en iyisini seçer, değerlendirir ve kaydeder.

Kullanım:
    python -m radar_classifier.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.tree import plot_tree

from .config import (
    CLASSES,
    FEATURES,
    FIGURES_DIR,
    METRICS_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)
from .data import clean, load_raw, save_clean
from .model import candidate_models, save_model


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Aday modelleri 5 katlı tabakalı çapraz doğrulama ile karşılaştırır."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in candidate_models().items():
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=["accuracy", "f1_macro"], return_train_score=True
        )
        rows.append(
            {
                "model": name,
                "cv_accuracy": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_f1_macro": scores["test_f1_macro"].mean(),
                "cv_f1_std": scores["test_f1_macro"].std(),
                "train_accuracy": scores["train_accuracy"].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)


def _plot_comparison(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    r = results.sort_values("cv_f1_macro")
    ax.barh(r["model"], r["cv_f1_macro"] * 100, xerr=r["cv_f1_std"] * 100,
            color="#4C72B0", capsize=4)
    for i, v in enumerate(r["cv_f1_macro"] * 100):
        ax.text(51, i, f"%{v:.1f}", va="center", ha="left", color="white",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("5 katlı CV makro F1 (%)")
    ax.set_xlim(50, 100)
    ax.set_title("Model Karşılaştırması")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_karsilastirma.png", dpi=120)
    plt.close(fig)


def _plot_confusion(y_test, y_pred) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=CLASSES, ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Karışıklık Matrisi (test verisi)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "karisiklik_matrisi.png", dpi=120)
    plt.close(fig)


def _plot_importance(model, X_test, y_test) -> dict:
    result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=RANDOM_STATE
    )
    imp = pd.Series(result.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(imp.index, imp.values, color="#55A868")
    ax.set_xlabel("Doğrulukta ortalama düşüş (permütasyon önemi)")
    ax.set_title("Özellik Önemi")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ozellik_onemi.png", dpi=120)
    plt.close(fig)
    return {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).items()}


def _plot_tree(X_train, y_train) -> None:
    """Yorumlanabilirlik için referans karar ağacını çizer."""
    tree = clone(candidate_models()["Karar Ağacı (depth=4)"]).fit(X_train, y_train)
    clf = tree.named_steps["clf"]
    fig = plt.figure(figsize=(20, 9))
    plot_tree(clf, feature_names=FEATURES, class_names=list(clf.classes_),
              filled=True, rounded=True, fontsize=9)
    plt.title("Karar Ağacı (açıklanabilir referans model)")
    fig.savefig(FIGURES_DIR / "karar_agaci.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def main() -> dict:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = clean(load_raw())
    save_clean(df)
    X, y = df[FEATURES], df[TARGET]
    print(f"Temiz veri: {df.shape[0]} satır, {len(FEATURES)} özellik")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 1) Model seçimi yalnızca eğitim verisi üzerinde çapraz doğrulamayla yapılır
    results = compare_models(X_train, y_train)
    print("\nÇapraz doğrulama sonuçları (eğitim verisi):")
    print(results.round(3).to_string(index=False))
    _plot_comparison(results)

    best_name = results.loc[0, "model"]
    best = clone(candidate_models()[best_name]).fit(X_train, y_train)
    print(f"\nSeçilen model: {best_name}")

    # 2) Test verisi yalnızca en sonda, bir kez kullanılır
    y_pred = best.predict(X_test)
    report = classification_report(y_test, y_pred, labels=CLASSES, output_dict=True)
    metrics = {
        "model": best_name,
        "egitim_tarihi": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "veri_satiri": len(df),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "train_accuracy": round(best.score(X_train, y_train), 4),
        # Savunma açısından en kritik metrik: drone'ların kaçının yakalandığı
        "mini_iha_recall": round(
            recall_score(y_test, y_pred, labels=["mini_iha"], average="macro"), 4
        ),
        "sinif_raporu": {
            c: {k: round(v, 4) for k, v in report[c].items()} for c in CLASSES
        },
        "cv_sonuclari": results.round(4).to_dict(orient="records"),
    }

    print("\n" + classification_report(y_test, y_pred, labels=CLASSES, digits=3))
    print(f"Test doğruluğu   : %{metrics['test_accuracy'] * 100:.1f}")
    print(f"Mini İHA duyarlılığı (recall): %{metrics['mini_iha_recall'] * 100:.1f}")

    _plot_confusion(y_test, y_pred)
    metrics["ozellik_onemi"] = _plot_importance(best, X_test, y_test)
    _plot_tree(X_train, y_train)

    # 3) Son model tüm veriyle yeniden eğitilip kaydedilir
    final = clone(candidate_models()[best_name]).fit(X, y)
    path = save_model(final, meta={k: metrics[k] for k in
                                   ("model", "egitim_tarihi", "test_accuracy",
                                    "test_f1_macro", "mini_iha_recall")})
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nModel kaydedildi : {path}")
    print(f"Metrikler        : {METRICS_PATH}")
    print(f"Grafikler        : {FIGURES_DIR}")
    return metrics


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    main()
