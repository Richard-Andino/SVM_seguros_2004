import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# CONFIGURACIÓN
# ============================================================
st.set_page_config(
    page_title="Risk Analytics System",
    page_icon="🛡️",
    layout="centered"
)

# ============================================================
# CSS (ESTÉTICA DARK MODERNA)
# ============================================================
st.markdown("""
<style>
    .stApp { background-color: #0b0e14; color: #e0e0e0; }
    .main-title { color: #00f2ff; text-align: center; font-size: 2.5rem; font-weight: 800; }
    .subtitle { color: #888; text-align: center; margin-bottom: 2rem; }
    .form-card { background: #161b22; padding: 2rem; border-radius: 15px; border: 1px solid #30363d; }
    .stButton > button { 
        width: 100%; background: #00f2ff; color: #000; font-weight: bold; border-radius: 8px; border: none; 
    }
    .result-box { background: #1c2128; padding: 2rem; border-radius: 15px; border: 1px solid #00f2ff; text-align: center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>🛡️ Risk Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistema de Clasificación Actuarial v2026</p>", unsafe_allow_html=True)

# ============================================================
# MODELOS (Lógica 100% intacta)
# ============================================================
MODEL_DIR = Path("models")
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
KMEANS_PATH = MODEL_DIR / "kmeans_riesgo_actuarial.pkl"

@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return preprocessor, kmeans

preprocessor, modelo = cargar_modelos()

# ============================================================
# FORMULARIO (Estructura intacta)
# ============================================================
with st.container():
    st.markdown("<div class='form-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", 18, 100, 30)
        sex = st.radio("Sexo", ["male", "female"], horizontal=True)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    with col2:
        children = st.number_input("Hijos", 0, 10, 0)
        smoker = st.radio("Fumador", ["yes", "no"], horizontal=True)
        region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])

    charges = st.number_input("Gastos médicos", 0, 100000, 5000)
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICCIÓN (Lógica de inferencia intacta)
# ============================================================
if st.button("🔍 Procesar Análisis"):
    cliente = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "charges": charges
    }])

    X_transformed = preprocessor.transform(cliente)
    cluster = modelo.predict(X_transformed)[0]

    interpretacion = {
        0: "Riesgo Bajo 🟢",
        1: "Riesgo Medio 🟡",
        2: "Riesgo Alto 🔴",
        3: "Riesgo Crítico ⚠️"
    }

    resultado = interpretacion.get(cluster, f"Cluster {cluster}")

    st.markdown(f"""
        <div class="result-box">
            <h3 style="color: #00f2ff;">Resultado del análisis</h3>
            <p style="font-size: 1.8rem; font-weight: bold;">{resultado}</p>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Ingrese los datos y presione el botón para comenzar.")

# ============================================================
# FOOTER
# ============================================================
st.markdown("<div class='subtitle' style='margin-top:2rem;'>© 2026 ISC - Angeles Euceda</div>", unsafe_allow_html=True)
