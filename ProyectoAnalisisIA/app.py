import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Análisis Exploratorio IA", layout="wide")

st.title("📊 Sistema Inteligente de Análisis Exploratorio de Datos")

# 1. Subir archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # 2. Mostrar estadísticas
    st.subheader("📈 Estadísticas Descriptivas")
    st.write(df.describe())

    # 3. Mostrar gráfico automático
    st.subheader("📊 Distribución de la primera columna numérica")

    columnas_numericas = df.select_dtypes(include='number').columns
    if len(columnas_numericas) > 0:
        col = columnas_numericas[0]
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribución: {col}")
        st.pyplot(fig)
    else:
        st.warning("No se encontraron columnas numéricas para graficar.")
else:
    st.info("Por favor, sube un archivo CSV para comenzar.")
