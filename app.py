import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================
st.set_page_config(
    page_title="Riesgo Actuarial IA",
    page_icon="📊",
    layout="centered"
)

# ============================================================
# ESTILOS CSS (MODERNO Y LIMPIO)
# ============================================================
st.markdown("""
<style>
    /* Fondo general */
    .stApp {
        background: linear-gradient(135deg, #0d47a1 0%, #1976d2 100%);
    }
    
    /* Contenedor principal */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    /* Títulos */
    .title { color: #0d47a1; text-align: center; font-weight: 700; }
    .subtitle { color: #555; text-align: center; margin-bottom: 2rem; }
    
    /* Etiquetas y texto de inputs */
    label { font-weight: 600 !important; color: #333 !important; }
    
    /* Resultado */
    .result-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border-left: 5px solid #1976d2;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CARGA DE MODELOS
# ============================================================
@st.cache_resource
def cargar_modelos():
    preprocessor = joblib.load("models/preprocessor.pkl")
    kmeans = joblib.load("models/kmeans_riesgo_actuarial.pkl")
    return preprocessor, kmeans

try:
    preprocessor, modelo = cargar_modelos()
except Exception as e:
    st.error("Error al cargar los modelos. Asegúrate de que la carpeta 'models' exista.")
    st.stop()

# ============================================================
# INTERFAZ
# ============================================================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.markdown("<h1 class='title'>📊 Clasificador de Riesgo Actuarial</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>IA-ISC • 2026</p>", unsafe_allow_html=True)

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

# ============================================================
# PREDICCIÓN
# ============================================================
if st.button("🔍 Calcular Riesgo", use_container_width=True):
    cliente = pd.DataFrame([{
        "age": age, "sex": sex, "bmi": bmi, 
        "children": children, "smoker": smoker, 
        "region": region, "charges": charges
    }])

    X_transformed = preprocessor.transform(cliente)
    cluster = modelo.predict(X_transformed)[0]

    interpretacion = {
        0: "🟢 Riesgo Bajo",
        1: "🟡 Riesgo Medio",
        2: "🔴 Riesgo Alto",
        3: "⚠️ Riesgo Crítico"
    }

    st.markdown(f"""
        <div class="result-box">
            <h3>Resultado del Análisis</h3>
            <h2 style="color:#1976d2;">{interpretacion.get(cluster, 'Desconocido')}</h2>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
