import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from radar_classifier.config import CLASSES, FEATURES, RANDOM_STATE, TARGET
from radar_classifier.data import clean, load_raw
from radar_classifier.model import candidate_models
from radar_classifier.predict import tahmin_et


@pytest.fixture(scope="module")
def veri():
    df = clean(load_raw())
    return train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET]
    )


@pytest.fixture(scope="module")
def model(veri):
    X_train, _, y_train, _ = veri
    return clone(candidate_models()["Random Forest"]).fit(X_train, y_train)


@pytest.mark.parametrize("isim", list(candidate_models()))
def test_tum_modeller_eksik_degerle_calisir(veri, isim):
    X_train, X_test, y_train, _ = veri
    m = clone(candidate_models()[isim]).fit(X_train, y_train)
    X = X_test.copy()
    X.iloc[0, 0] = np.nan
    assert len(m.predict(X)) == len(X)


def test_model_basarisi_makul(veri, model):
    _, X_test, _, y_test = veri
    assert model.score(X_test, y_test) >= 0.85


def test_tahmin_et_cikti_formati(model):
    t = tahmin_et(model, 350, 5000, 12.0, 0.8, 0)
    assert t.tur in CLASSES
    assert 0 <= t.guven <= 100
    assert abs(sum(t.olasiliklar.values()) - 100) < 1e-6
    assert t.tur == "helikopter_ucak"


def test_bariz_kus(model):
    t = tahmin_et(model, 55, 300, 0.008, 9.0, 0)
    assert t.tur == "kus"
