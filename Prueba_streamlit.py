import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Aplicación de prueba con Streamlit")

# Descripción
st.write("""
    Esta es una aplicación simple de prueba para demostrar cómo cargar datos
    y visualizar un gráfico en Streamlit.
""")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

# Si se ha subido un archivo
if uploaded_file is not None:
    # Leer el archivo CSV
    df = pd.read_csv(uploaded_file)

    # Mostrar los primeros 5 registros del archivo
    st.write("Vista previa de los primeros 5 registros del archivo:")
    st.dataframe(df.head())

    # Seleccionar columnas para graficar
    if len(df.columns) >= 2:
        columna_x = st.selectbox("Selecciona la columna para el eje X", df.columns)
        columna_y = st.selectbox("Selecciona la columna para el eje Y", df.columns)

        # Mostrar gráfico
        st.write(f"Gráfico de {columna_y} vs {columna_x}")
        fig, ax = plt.subplots()
