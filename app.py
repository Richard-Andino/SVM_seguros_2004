import streamlit as st
import pandas as pd
import joblib

# Configuración de página
st.set_page_config(page_title="Riesgo Actuarial AI", layout="centered")

# Estilos CSS para centrado absoluto y diseño moderno
st.markdown("""
<style>
    /* Fondo con gradiente */
    .stApp { background: linear-gradient(135deg, #0f2027, #203a43, #2c5364); }
    
    /* Contenedor centralizado */
    .central-card {
        background: rgba(255, 255, 255, 0.98);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.5);
        margin-top: 50px;
    }
    
    h1 { color: #1a237e; text-align: center; font-weight: 800; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3em; 
        background-color: #1a237e !important; color: white !important; 
        font-weight: bold; border: none; 
    }
</style>
""", unsafe_allow_html=True)

# Carga de modelos
@st.cache_resource
def cargar_modelos():
    return joblib.load("models/preprocessor.pkl"), joblib.load("models/kmeans_riesgo_actuarial.pkl")

preprocessor, modelo = cargar_modelos()

# Estructura central
st.markdown("<div class='central-card'>", unsafe_allow_html=True)
st.title("📊 Riesgo Actuarial IA")
st.caption("Sistema de Clasificación Profesional | 2026")

# Formulario en una sola columna lógica
age = st.number_input("Edad", 18, 100, 30)
col_a, col_b = st.columns(2)
with col_a:
    sex = st.selectbox("Sexo", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
with col_b:
    smoker = st.selectbox("¿Fumador?", ["yes", "no"])
    children = st.number_input("Hijos", 0, 10, 0)

region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
charges = st.number_input("Gastos médicos", 0, 100000, 5000)

st.write("")
if st.button("PROCESAR ANÁLISIS"):
    cliente = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children, 
                             "smoker": smoker, "region": region, "charges": charges}])
    
    cluster = modelo.predict(preprocessor.transform(cliente))[0]
    
    colores = {0: "#2e7d32", 1: "#fbc02d", 2: "#d32f2f", 3: "#8e24aa"}
    nombres = {0: "RIESGO BAJO", 1: "RIESGO MEDIO", 2: "RIESGO ALTO", 3: "RIESGO CRÍTICO"}
    
    st.markdown(f"""
        <div style="text-align:center; padding: 20px; border: 2px solid {colores[cluster]}; 
                    border-radius: 15px; background: #fdfdfd; margin-top: 20px;">
            <p style="margin:0; font-weight:bold; color:{colores[cluster]}; font-size: 1.5rem;">
                {nombres[cluster]}
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
