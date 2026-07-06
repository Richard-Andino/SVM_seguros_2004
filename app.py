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
# CSS (EDITORIAL PAPER — MINIMAL CASUAL)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ─── FONDO CÁLIDO MINIMAL ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #f4f3ef !important;
    font-family: 'DM Sans', sans-serif;
    color: #1c1917;
}

[data-testid="stAppViewContainer"] {
    background: #f4f3ef !important;
}

/* ─── TÍTULO ─── */
.main-title {
    text-align: center;
    color: #1c1917;
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.2rem;
    text-shadow: none;
}

.deco-line {
    width: 32px;
    height: 1px;
    background: #b45309;
    margin: 0.7rem auto;
    opacity: 0.5;
}

.subtitle {
    text-align: center;
    color: #a8a29e;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 400;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.description {
    text-align: center;
    color: #78716c;
    font-size: 0.95rem;
    margin-bottom: 2.5rem;
}

/* ─── TÍTULO "DATOS DEL CLIENTE" ─── */
.form-title-outside {
    color: #78716c;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 500;
    margin-bottom: 0.6rem;
    text-align: left;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─── CUADRO DEL FORMULARIO ─── */
.form-card {
    background: #ffffff;
    padding: 2.5rem;
    border-radius: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #e8e6e1;
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
    color: #57534e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Texto de los radio buttons */
.stRadio [role="radiogroup"] label,
.stRadio [data-testid="stMarkdownContainer"] p,
.stRadio span[data-baseweb="radio"] div,
.stRadio div[role="radio"] {
    color: #44403c !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 400 !important;
}

/* Opciones del selectbox */
.stSelectbox [data-testid="stMarkdownContainer"] p,
.stSelectbox div[data-baseweb="select"] span {
    color: #44403c !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Input numbers */
.stNumberInput input {
    color: #1c1917 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    background: #fafaf9 !important;
    border: 1px solid #d6d3cd !important;
    border-radius: 6px !important;
}

.stNumberInput input:focus {
    border-color: #b45309 !important;
    box-shadow: 0 0 0 3px rgba(180,83,9,0.08) !important;
}

.stNumberInput input::placeholder {
    color: #a8a29e !important;
}

/* Selectbox container */
div[data-baseweb="select"] > div {
    background: #fafaf9 !important;
    border: 1px solid #d6d3cd !important;
    border-radius: 6px !important;
    color: #1c1917 !important;
}

/* ─── BOTÓN PREDECIR ─── */
.stButton > button {
    background: #1c1917 !important;
    color: #f4f3ef !important;
    border-radius: 6px !important;
    padding: 0.8rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border: none !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: #292524 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.14) !important;
    transform: translateY(-1px) !important;
}

/* ─── RESULTADO ─── */
.result-card {
    background: #ffffff;
    padding: 2.5rem;
    border-radius: 10px;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
    border: 1px solid #e8e6e1;
    animation: fadeInUp 0.4s ease;
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: #b45309;
    opacity: 0.6;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-label {
    color: #a8a29e;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.cluster-badge {
    display: inline-block;
    padding: 0.6rem 2rem;
    border-radius: 6px;
    background: #1c1917;
    color: #f4f3ef;
    font-family: 'Playfair Display', serif;
    font-weight: 600;
    font-size: 1.4rem;
    border: none;
    box-shadow: none;
    letter-spacing: 0.01em;
    margin-top: 0.3rem;
}

/* ─── INFO BOX ─── */
.stAlert {
    background: #fafaf9 !important;
    color: #78716c !important;
    border: 1px solid #e8e6e1 !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}

//* ─── FOOTER ─── */
.footer-text {
    text-align: center;
    margin-top: 3rem;
    color: #c8c4bd;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
}

/* ─── DIVIDER ─── */
hr {
    border: none;
    height: 1px;
    background: #e8e6e1;
    margin: 2rem 0;
}

/* ─── GENERAL TEXT ─── */
.stMarkdown p, .stMarkdown div {
    color: #44403c;
}y: 'DM Sans', sans-serif;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    margin-top: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #ede9e3;
}

.section-sep {
    height: 1px;
    background: #f0ede8;
    margin: 1.4rem 0 1.0rem;
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
st.markdown("<p class='subtitle'>IA-ISC &nbsp;&middot;&nbsp; Angeles Euceda &nbsp;&middot;&nbsp; 20221930061 &nbsp;&middot;&nbsp; 2026</p>", unsafe_allow_html=True)



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
st.markdown("<div class='form-title-outside'>Datos del cliente</div>", unsafe_allow_html=True)

# ============================================================
# FORMULARIO EN CUADRO BLANCO (sin título adentro)
# ============================================================
with st.container():

    # ── SECCIÓN 1: Perfil del asegurado
    st.markdown("<div class='section-header'>// perfil del asegurado</div>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1.0, 1.1, 1.3])
    with col_a:
        age = st.number_input("Edad", 18, 100, 42)
    with col_b:
        sex = st.radio("Sexo", ["male", "female"], horizontal=True)
    with col_c:
        region = st.selectbox(
            "Región",
            ["southeast", "southwest", "northeast", "northwest"],
            index=2
        )

    st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

    # ── SECCIÓN 2: Indicadores de salud
    st.markdown("<div class='section-header'>// indicadores de salud</div>", unsafe_allow_html=True)
    col_d, col_e = st.columns([1.3, 1])
    with col_d:
        bmi = st.number_input("BMI", 10.0, 60.0, 31.5)
    with col_e:
        smoker = st.radio(
            "Fumador",
            ["yes", "no"],
            horizontal=True
        )

    st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

    # ── SECCIÓN 3: Cobertura
    st.markdown("<div class='section-header'>// cobertura</div>", unsafe_allow_html=True)
    col_f, col_g = st.columns([1, 2.2])
    with col_f:
        children = st.number_input("Hijos", 0, 10, 2)
    with col_g:
        charges = st.number_input("Gastos médicos", 0, 100000, 18750)

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
            <div class="result-label">Resultado del análisis</div>
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
Sistema de Riesgo Actuarial con IA &nbsp;&middot;&nbsp; ISC 2026
</div>
""", unsafe_allow_html=True)   
