import streamlit as st
import pandas as pd
import numpy as np

# Configurar la página
st.set_page_config(
    page_title="Forecasting ML",
    page_icon="📊",
    layout="wide"
)

# Encabezado
st.title("📊 Aplicación de Forecasting con Machine Learning")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una página:",
    ["Home", "Análisis de Datos", "Predicciones", "Modelos"]
)

# Página Home
if page == "Home":
    st.header("Bienvenido")
    st.write("""
    Esta es una aplicación de forecasting construida con Streamlit y Machine Learning.
    
    **Características:**
    - 📈 Análisis exploratorio de datos
    - 🤖 Modelos de predicción
    - 📊 Visualizaciones interactivas
    """)

# Página Análisis de Datos
elif page == "Análisis de Datos":
    st.header("Análisis de Datos")
    st.write("Sube un archivo CSV para análisis exploratorio")
    
    uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Vista previa de los datos:**")
        st.dataframe(df.head())
        
        st.write(f"**Forma del dataset:** {df.shape}")
        st.write(f"**Información del dataset:**")
        st.dataframe(df.describe())

# Página Predicciones
elif page == "Predicciones":
    st.header("Predicciones")
    st.write("Funcionalidad de predicciones en desarrollo...")

# Página Modelos
elif page == "Modelos":
    st.header("Gestión de Modelos")
    st.write("Funcionalidad de modelos en desarrollo...")

# Footer
st.markdown("---")
st.markdown("© 2026 - Proyecto de Forecasting")
