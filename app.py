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
# CSS (DARK MEDICAL INTELLIGENCE — TEAL + DARK)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ─── FONDO OSCURO TÉCNICO ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #060b18 !important;
    font-family: 'JetBrains Mono', monospace;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 50% -5%, rgba(0,212,170,0.10) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,212,170,0.035) 39px, rgba(0,212,170,0.035) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,212,170,0.035) 39px, rgba(0,212,170,0.035) 40px),
        #060b18 !important;
}

/* ─── TÍTULO ─── */
.main-title {
    text-align: center;
    color: #00d4aa;
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 40px rgba(0,212,170,0.45);
}

.deco-line {
    width: 50px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4aa, transparent);
    margin: 0.6rem auto;
}

.subtitle {
    text-align: center;
    color: rgba(0,212,170,0.5);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.description {
    text-align: center;
    color: rgba(226,232,240,0.6);
    font-size: 0.9rem;
    margin-bottom: 2.5rem;
}

/* ─── TÍTULO "DATOS DEL CLIENTE" ─── */
.form-title-outside {
    color: rgba(0,212,170,0.9);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
    text-align: left;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    border-left: 2px solid #00d4aa;
    padding-left: 0.75rem;
}

/* ─── CUADRO DEL FORMULARIO ─── */
.form-card {
    background: rgba(8, 15, 32, 0.85);
    padding: 2.5rem;
    border-radius: 4px;
    box-shadow: 0 0 0 1px rgba(0,212,170,0.15), 0 24px 80px rgba(0,0,0,0.6);
    border: 1px solid rgba(0,212,170,0.2);
    backdrop-filter: blur(20px);
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
    color: rgba(0,212,170,0.85) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.70rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* Texto de los radio buttons */
.stRadio [role="radiogroup"] label,
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio span[data-baseweb="radio"] div,
.stRadio div[role="radio"] {
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Opciones del selectbox */
.stSelectbox [data-testid="stMarkdownContainer"] p,
.stSelectbox div[data-baseweb="select"] span {
    color: #cbd5e1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Input numbers */
.stNumberInput input {
    color: #00d4aa !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    background: rgba(0,212,170,0.04) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    border-radius: 2px !important;
}

.stNumberInput input:focus {
    border-color: #00d4aa !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.12) !important;
}

.stNumberInput input::placeholder {
    color: rgba(226,232,240,0.25) !important;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
    background: rgba(0,212,170,0.04) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    border-radius: 2px !important;
    color: #cbd5e1 !important;
}

/* ─── BOTÓN PREDECIR ─── */
.stButton > button {
    background: #00d4aa !important;
    color: #060b18 !important;
    border-radius: 2px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    border: none !important;
    box-shadow: 0 0 30px rgba(0,212,170,0.28) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #00f0c2 !important;
    box-shadow: 0 0 55px rgba(0,212,170,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ─── RESULTADO ─── */
.result-card {
    background: rgba(8, 15, 32, 0.92);
    padding: 2.5rem;
    border-radius: 4px;
    text-align: center;
    margin-top: 2rem;
    box-shadow: 0 0 0 1px rgba(0,212,170,0.2), 0 24px 80px rgba(0,0,0,0.5);
    border: 1px solid rgba(0,212,170,0.3);
    animation: fadeInUp 0.45s ease;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4aa 40%, transparent);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(15px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-label {
    color: rgba(0,212,170,0.6);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.cluster-badge {
    display: inline-block;
    padding: 0.65rem 2.2rem;
    border-radius: 2px;
    background: transparent;
    color: #00d4aa;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.55rem;
    border: 1px solid #00d4aa;
    box-shadow: 0 0 22px rgba(0,212,170,0.2), inset 0 0 20px rgba(0,212,170,0.05);
    letter-spacing: 0.04em;
    margin-top: 0.5rem;
}

/* ─── INFO BOX ─── */
.stAlert {
    background: rgba(0,212,170,0.05) !important;
    color: rgba(226,232,240,0.75) !important;
    border: 1px solid rgba(0,212,170,0.18) !important;
    border-radius: 2px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* ─── FOOTER ─── */
.footer-text {
    text-align: center;
    margin-top: 3rem;
    color: rgba(0,212,170,0.28);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

/* ─── DIVIDER ─── */
hr {
    border: none;
    height: 1px;
    background: rgba(0,212,170,0.1);
    margin: 2rem 0;
}

/* ─── GENERAL TEXT ─── */
.stMarkdown p, .stMarkdown div {
    color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("<h1 class='main-title'>Clasificador de Riesgo Actuarial</h1>", unsafe_allow_html=True)
st.markdown("<div class='deco-line'></div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Richard Andino &nbsp;&middot;&nbsp; 20231900184 &nbsp;&middot;&nbsp</p>", unsafe_allow_html=True)



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
st.markdown("<div class='form-title-outside'>// datos del cliente</div>", unsafe_allow_html=True)

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
            <div class="result-label">// análisis completado</div>
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
sistema de riesgo actuarial &nbsp;&middot;&nbsp; ia-isc 2026
</div>
""", unsafe_allow_html=True)   
