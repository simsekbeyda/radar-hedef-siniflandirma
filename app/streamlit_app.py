"""Radar operatörü için web arayüzü.

Çalıştırma:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from radar_classifier.config import (
    ACIKLAMALAR,
    FIGURES_DIR,
    GUVEN_ESIGI,
    MODEL_PATH,
)
from radar_classifier.model import load_model
from radar_classifier.predict import tahmin_et

st.set_page_config(page_title="Radar Hedef Sınıflandırma", layout="wide")


@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        from radar_classifier.train import main as train

        with st.spinner("Model ilk kez eğitiliyor..."):
            train()
    return load_model()


model, meta = get_model()

st.title("Radar Hedef Sınıflandırma")
st.caption(
    "Radar ölçümlerinden hava hedefinin türünü tahmin eden makine öğrenmesi modeli. "
    "Veri sentetiktir; eğitim/demo amaçlıdır."
)

with st.sidebar:
    st.header("Radar ölçümleri")
    hiz = st.slider("Hız (km/saat)", 0.0, 500.0, 45.0, 1.0)
    irtifa = st.slider("İrtifa (metre)", 0.0, 10000.0, 150.0, 10.0)
    rcs = st.number_input(
        "Radar kesit alanı (m²)", min_value=0.0, max_value=50.0, value=0.03,
        step=0.001, format="%.4f",
    )
    manevra = st.slider("Manevra skoru (yön değişimi / dk)", 0.0, 15.0, 2.5, 0.1)
    pervane = st.radio("Pervane imzası (mikro-Doppler)", ["Var", "Yok"], horizontal=True)

    st.divider()
    st.markdown(
        f"**Model:** {meta.get('model', '-')}  \n"
        f"**Test doğruluğu:** %{meta.get('test_accuracy', 0) * 100:.1f}  \n"
        f"**Drone yakalama (recall):** %{meta.get('mini_iha_recall', 0) * 100:.1f}"
    )

t = tahmin_et(model, hiz, irtifa, rcs, manevra, 1 if pervane == "Var" else 0)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Sonuç")
    st.metric("Tahmin edilen hedef", t.ad, f"Güven: %{t.guven:.0f}", delta_color="off")
    if t.tur == "mini_iha":
        st.error(f"{t.oneri}")
    elif t.tur == "kus":
        st.success(f"{t.oneri}")
    else:
        st.info(f"{t.oneri}")
    if t.kontrol_gerekli:
        st.warning(
            f"Modelin güveni %{GUVEN_ESIGI:.0f}'in altında. Sonucu bir operatör kontrol etmeli."
        )

with col2:
    st.subheader("Sınıf olasılıkları")
    proba = pd.DataFrame(
        {"Hedef": [ACIKLAMALAR[c][0] for c in t.olasiliklar],
         "Olasılık (%)": list(t.olasiliklar.values())}
    ).set_index("Hedef")
    st.bar_chart(proba)

st.divider()
st.subheader("Model performansı")
figs = [
    ("Model karşılaştırması", "model_karsilastirma.png"),
    ("Karışıklık matrisi", "karisiklik_matrisi.png"),
    ("Özellik önemi", "ozellik_onemi.png"),
]
cols = st.columns(len(figs))
for c, (baslik, dosya) in zip(cols, figs):
    yol = FIGURES_DIR / dosya
    if yol.exists():
        c.image(str(yol), caption=baslik, width="stretch")
