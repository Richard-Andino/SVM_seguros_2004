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
    page_title="Riesgo Actuarial IA",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS (FONDO AZUL + FORMULARIO EN CUADRO BLANCO)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* ─── FONDO AZUL DE TODA LA PÁGINA ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 40%, #42a5f5 100%) !important;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0d47a1 0%, #1976d2 40%, #42a5f5 100%) !important;
}

/* ─── TÍTULO ─── */
.main-title {
    text-align: center;
    color: #ffffff;
    font-size: 2.8rem;
    font-weight: 700;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3);
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #bbdefb;
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 2rem;
}

.description {
    text-align: center;
    color: #e3f2fd;
    font-size: 1rem;
    margin-bottom: 2.5rem;
}

/* ─── TÍTULO "DATOS DEL CLIENTE" FUERA DEL CUADRO ─── */
.form-title-outside {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    text-align: center;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* ─── CUADRO DEL FORMULARIO (TARJETA BLANCA) ─── */
.form-card {
    background: rgba(255, 255, 255, 0.95);
    padding: 2.5rem;
    border-radius: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    border: 2px solid rgba(255,255,255,0.3);
    backdrop-filter: blur(10px);
    margin-bottom: 2rem;
}

/* Labels de todos los inputs */
.stNumberInput label,
.stRadio label,
.stSelectbox label,
.stNumberInput > div > label,
.stRadio > div > label,
.stSelectbox > div > label,
[data-testid="stNumberInput"] label,
[data-testid="stRadio"] label,
[data-testid="stSelectbox"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* Texto de los radio buttons (male, female, yes, no) */
.stRadio [role="radiogroup"] label,
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio span[data-baseweb="radio"] div,
.stRadio div[role="radio"] {
    color: #ffffff !important;
    font-weight: 600 !important;
}

/* Opciones del selectbox */
.stSelectbox [data-testid="stMarkdownContainer"] p,
.stSelectbox div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Input numbers */
.stNumberInput input {
    color: #333333 !important;
    font-weight: 600 !important;
    background: #f5f9ff !important;
    border: 2px solid #bbdefb !important;
    border-radius: 12px !important;
}

/* Placeholder y texto dentro de inputs */
.stNumberInput input::placeholder {
    color: #999999 !important;
}

/* ─── BOTÓN PREDECIR ─── */
.stButton > button {
    background: linear-gradient(135deg, #1976d2, #0d47a1) !important;
    color: white !important;
    border-radius: 14px !important;
    padding: 0.8rem 2rem !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(13, 71, 161, 0.4) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #0d47a1, #1565c0) !important;
    box-shadow: 0 12px 35px rgba(13, 71, 161, 0.6) !important;
    transform: translateY(-2px) !important;
}

/* ─── RESULTADO ─── */
.result-card {
    background: linear-gradient(135deg, #ffffff, #e3f2fd);
    padding: 2.5rem;
    border-radius: 24px;
    text-align: center;
    margin-top: 2rem;
    box-shadow: 0 15px 50px rgba(0,0,0,0.25);
    border: 2px solid rgba(255,255,255,0.5);
    animation: fadeInUp 0.6s ease;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.cluster-badge {
    display: inline-block;
    padding: 0.8rem 2rem;
    border-radius: 50px;
    background: linear-gradient(135deg, #1976d2, #0d47a1);
    color: white;
    font-weight: bold;
    font-size: 1.4rem;
    box-shadow: 0 6px 20px rgba(13, 71, 161, 0.4);
    margin-top: 1rem;
}

/* ─── INFO BOX ─── */
.stAlert {
    background: rgba(255,255,255,0.15) !important;
    color: #e3f2fd !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    border-radius: 16px !important;
}

/* ─── FOOTER ─── */
.footer-text {
    text-align: center;
    margin-top: 3rem;
    color: rgba(255,255,255,0.7);
    font-size: 0.9rem;
}

/* ─── DIVIDER ─── */
hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.2);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>📊 Clasificador de Riesgo Actuarial</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA-ISC • Angeles Euceda • 20221930061 • 2026</p>", unsafe_allow_html=True)



# ============================================================
# MODELOS
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
# TÍTULO DEL FORMULARIO AFUERA DEL CUADRO
# ============================================================
st.markdown("<div class='form-title-outside'>🧾 Datos del cliente</div>", unsafe_allow_html=True)

# ============================================================
# FORMULARIO EN CUADRO BLANCO (sin título adentro)
# ============================================================
with st.container():
    

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad", 18, 100, 30)

        sex = st.radio(
            "Sexo",
            ["male", "female"],
            horizontal=True
        )

        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

    with col2:
        children = st.number_input("Hijos", 0, 10, 0)

        smoker = st.radio(
            "Fumador",
            ["yes", "no"],
            horizontal=True
        )

        region = st.selectbox(
            "Región",
            ["southeast", "southwest", "northeast", "northwest"]
        )

    charges = st.number_input("Gastos médicos", 0, 100000, 5000)

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PREDICCIÓN
# ============================================================
if st.button("🔍 Predecir riesgo"):

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
        <div class="result-card">
            <div style="font-size:3.5rem; margin-bottom:0.5rem;">📊</div>
            <div style="color:#0d47a1; font-size:1.2rem; font-weight:600; margin-bottom:0.5rem;">Resultado del análisis</div>
            <div class="cluster-badge">{resultado}</div>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Ingrese los datos del cliente y presione **Predecir riesgo**")

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer-text">
📊 Sistema de Riesgo Actuarial con IA • ISC 2026
</div>
""", unsafe_allow_html=True)   
