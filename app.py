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
    page_title="Predicción de Valor Cliente",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS (FONDO VIOLETA + TARJETA MODERNA)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #4a148c 0%, #7b1fa2 50%, #d500f9 100%) !important;
    font-family: 'Inter', sans-serif;
}

.main-title { text-align: center; color: #ffffff; font-weight: 700; margin-bottom: 0.5rem; }
.subtitle { text-align: center; color: #e1bee7; font-size: 1.1rem; }
.form-title-outside { color: #ffffff; font-size: 1.4rem; font-weight: 600; text-align: center; margin: 1rem 0; }

.stButton > button {
    background: #ffffff !important;
    color: #4a148c !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    width: 100% !important;
    border: none !important;
}

.result-card {
    background: #ffffff;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>💎 Segmentación de Clientes</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA-Analytics • Estrategia Comercial • 2026</p>", unsafe_allow_html=True)

# ============================================================
# CARGA DE MODELOS (Asume archivos renombrados si es necesario)
# ============================================================
MODEL_DIR = Path("models")
@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    modelo = joblib.load(MODEL_DIR / "kmeans_clv.pkl") # Cambiado el nombre del archivo
    return preprocessor, modelo

preprocessor, modelo = cargar_modelos()

# ============================================================
# FORMULARIO
# ============================================================
st.markdown("<div class='form-title-outside'>⚙️ Parámetros de Fidelización</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    tenure = st.number_input("Meses como cliente", 0, 120, 12)
    frecuencia = st.number_input("Compras anuales", 0, 50, 5)
with col2:
    gasto_promedio = st.number_input("Ticket promedio ($)", 0, 10000, 100)
    canal = st.selectbox("Canal de Adquisición", ["Social", "Email", "Direct", "Referido"])

# ============================================================
# PREDICCIÓN
# ============================================================
if st.button("🚀 Analizar Segmento"):
    # Asegúrate de que las columnas coincidan con el entrenamiento
    input_data = pd.DataFrame([{
        "tenure": tenure,
        "frequency": frecuencia,
        "avg_spend": gasto_promedio,
        "channel": canal
    }])

    X_transformed = preprocessor.transform(input_data)
    cluster = modelo.predict(X_transformed)[0]

    segmentos = {
        0: "Clientes Ocasionales 🥉",
        1: "Potencial de Crecimiento 🥈",
        2: "Clientes Premium 🥇",
        3: "Clientes VIP 💎"
    }

    st.markdown(f"""
        <div class="result-card">
            <h3>Segmento Identificado:</h3>
            <h2 style="color:#4a148c;">{segmentos.get(cluster, 'Cliente Estándar')}</h2>
        </div>
    """, unsafe_allow_html=True)
