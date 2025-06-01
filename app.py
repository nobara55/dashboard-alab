import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

st.title("📈 Dashboard Financiero ALAB")
st.write("Datos extraídos de Yahoo Finance")

# Entrada del usuario
ticker = st.text_input("Ingrese el ticker de la acción", value="AAPL")
period = st.selectbox("Seleccione período", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"])

# Descargar datos
data = yf.download(ticker, period=period)

if not data.empty:
    # Gráfico de cierre
    fig = px.line(data, x=data.index, y='Close', title=f"Precio de cierre de {ticker.upper()}")
    st.plotly_chart(fig)

    # Mostrar tabla
    st.subheader("Datos históricos recientes")
    st.dataframe(data.tail(10))
else:
    st.warning("No se encontraron datos. Verifica el ticker.")
