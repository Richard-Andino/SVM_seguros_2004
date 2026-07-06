import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuración de página
st.set_page_config(page_title="Actuarial Risk AI", layout="wide")

# Estilos CSS Modernos
st.markdown("""
<style>
    /* Fondo estilo Glassmorphism */
    .stApp { background: #f0f2f6; }
    
    .sidebar .sidebar-content { background: #ffffff; }
    
    .main-card {
        background: white;
        padding: 3rem;
        border-radius: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        text-align: center;
        margin: 2rem auto;
        max-width: 800px;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: #1a237e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Carga de modelos
@st.cache_resource
def cargar_modelos():
    return joblib.load("models/preprocessor.pkl"), joblib.load("models/kmeans_riesgo_actuarial.pkl")

preprocessor, modelo = cargar_modelos()

# --- BARRA LATERAL (ENTRADA DE DATOS) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.header("⚙️ Configuración del Cliente")
    
    age = st.slider("Edad", 18, 100, 30)
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    children = st.number_input("Número de Hijos", 0, 10, 0)
    smoker = st.radio("¿Es fumador?", ["yes", "no"])
    region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
    charges = st.number_input("Gastos médicos esperados", 0, 100000, 5000)
    
    submit = st.button("ANALIZAR RIESGO", use_container_width=True)

# --- ÁREA CENTRAL (VISUALIZACIÓN) ---
st.title("📊 Actuarial Risk Intelligence")
st.markdown("---")

if submit:
    cliente = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children, 
                             "smoker": smoker, "region": region, "charges": charges}])
    
    cluster = modelo.predict(preprocessor.transform(cliente))[0]
    
    risks = {
        0: ("🟢 Riesgo Bajo", "Perfil financiero estable y saludable."),
        1: ("🟡 Riesgo Medio", "Se requiere monitoreo preventivo."),
        2: ("🔴 Riesgo Alto", "Alta probabilidad de siniestralidad."),
        3: ("⚠️ Riesgo Crítico", "Requiere intervención inmediata.")
    }
    
    label, desc = risks.get(cluster, ("Desconocido", ""))
    
    st.markdown(f"""
    <div class="main-card">
        <h3>Resultado de la Clasificación</h3>
        <div class="metric-value">{label}</div>
        <p style="font-size: 1.2rem; color: #555;">{desc}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Ajuste los parámetros en el panel lateral y presione el botón para iniciar el análisis.")
