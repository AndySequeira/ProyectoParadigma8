import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(page_title="Análisis Exploratorio IA", layout="wide")
st.title("📊 Sistema Inteligente de Análisis Exploratorio de Datos")

# 1. Subir archivo
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("🔍 Vista previa de los datos")
    st.dataframe(df.head())

    # 2. Estadísticas
    st.subheader("📈 Estadísticas Descriptivas")
    st.write(df.describe(include='all'))

    # 3. Tipos de variables
    st.subheader("🔢 Tipos de variables")
    st.write(df.dtypes)

    # 4. Distribuciones de todas las columnas numéricas
    columnas_numericas = df.select_dtypes(include='number').columns
    if len(columnas_numericas) > 0:
        st.subheader("📊 Histogramas de variables numéricas")
        for col in columnas_numericas:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribución: {col}")
            st.pyplot(fig)
    else:
        st.warning("No se encontraron columnas numéricas para graficar.")

    # 5. Matriz de correlación
    if len(columnas_numericas) >= 2:
        st.subheader("🔗 Matriz de Correlación")
        corr = df[columnas_numericas].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # 6. Detección de valores atípicos
    st.subheader("🚨 Detección de outliers (z-score > 3)")
    outliers = {}
    for col in columnas_numericas:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers[col] = df[z_scores > 3][col]
    st.write("Valores atípicos detectados:")
    st.write({k: v.dropna().tolist() for k, v in outliers.items() if not v.empty})

    # 7. Clustering automático
    st.subheader("🧠 Agrupamiento automático con KMeans")
    num_clusters = st.slider("Selecciona número de clusters (K)", 2, 6, 3)
    scaler = StandardScaler()
    datos_scaled = scaler.fit_transform(df[columnas_numericas].dropna())
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    etiquetas = kmeans.fit_predict(datos_scaled)

    df_clusters = df.copy()
    df_clusters["Cluster"] = etiquetas
    st.write(df_clusters.head())

    # 8. Visualización 2D de los clusters
    if len(columnas_numericas) >= 2:
        st.subheader("🎯 Visualización de Clusters")
        fig, ax = plt.subplots()
        sns.scatterplot(
            x=datos_scaled[:, 0], y=datos_scaled[:, 1],
            hue=etiquetas, palette="tab10", s=100, ax=ax
        )
        ax.set_xlabel(columnas_numericas[0])
        ax.set_ylabel(columnas_numericas[1])
        st.pyplot(fig)

    # 9. Descargar datos con cluster
    st.subheader("⬇️ Descargar CSV con resultados")
    csv_resultado = df_clusters.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV con clusters",
        data=csv_resultado,
        file_name="resultado_clusters.csv",
        mime="text/csv"
    )

else:
    st.info("Por favor, sube un archivo CSV para comenzar.")

