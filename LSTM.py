# importacion de librerias

import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import streamlit as st

plt.style.use("seaborn-darkgrid")

warnings.filterwarnings("ignore")

st.markdown("# Modelo Predictivo - Red de memoria a corto plazo largo(LSTM)")
st.sidebar.header("LSTM")
st.write(
    """En esta página se podrá ver cómo funciona un modelo LSTM en la predicción del mercado de valores"""
)

ticker = st.text_input("Etiqueta de cotización", "ETH-USD")
st.write("La etiqueta de cotización actual es", ticker)

fecha_hoy = pd.Timestamp.today().strftime('%Y-%m-%d') #obtencion de la fecha de hoy

df = yf.download(ticker, start="2015-01-01", end=fecha_hoy )

st.write(df)

texto = "Data histórica de la acción " + ticker + " a la fecha " + fecha_hoy
st.write(texto)


# Graficar el historico del valor de cierre
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Close"], color="black", label="valor de cierre")
ax.set_title("historico del valor de cierre de la accion")
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor de cierre")
ax.legend()
st.pyplot(fig)

#obtengo la diferencia de open y close
df['Open-Close'] = df.Open - df.Close
df['High-Low'] = df.High - df.Low # tambien la de high y low que es mayor y menor precio de la accion en ese dia


X = df[['Open-Close','High-Low']] #de lo anterior obtengo mis datos de input
y = np.where(df['Close'].shift(-1) > df['Close'],1,0)

model = tf.keras.models.load_model('modelos/modelo_LSTM.h5')


#ajustando los datos para las predicciones LSTM
length = 10
# Tamaño del lote
batch_size = 32
# Crear el generador de secuencias para hacer predicciones
generator2 = TimeseriesGenerator(X, np.zeros((len(X),)), length=length, batch_size=batch_size)


# Hacer predicciones con el modelo
predicciones = model.predict(generator2)
predicciones_array = np.squeeze(predicciones)
umbral = 0.5
predicciones_binarias = (predicciones_array >= umbral).astype(int)

#ajustando el df para adjuntar las predicciones
datos_perdidos = len(y) - len(predicciones_binarias)
df = df[datos_perdidos:]


df['Predicted_Signal'] = predicciones_binarias #predigo los datos del dataframe X que antes dividi en entrenamiento y test


# Calculate daily returns
df['Return'] = df.Close.pct_change() #funcion para sacar el porcentaje de aumento o disminucion
# Calculate strategy returns
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)


# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()
st.write("Dataframe con retornos acumulativos")
df

# Plot Strategy Cumulative returns
df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
st.write("Dataframe con retornos de estrategia acumulativos")
df



st.write("Gráfica de los retornos de la estrategia vs. los retornos originales")
# Plot de los retornos acumulativos de la estrategia
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Cum_Ret"], color="red", label="Retornos Originales")
ax.plot(df["Cum_Strategy"], color="blue", label="Retornos de Estrategia")
ax.set_title("Retornos Acumulativos de la Estrategia vs. Retornos Originales")
ax.set_xlabel("Fecha")
ax.set_ylabel("Retornos Acumulativos")
ax.legend()
st.pyplot(fig)
