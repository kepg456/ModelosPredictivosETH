import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import warnings
import streamlit as st


plt.style.use("seaborn-darkgrid")

warnings.filterwarnings("ignore")

st.markdown("# Modelo Predictivo - Red Neural Convolucional")
st.sidebar.header("CONV")
st.write(
    """En esta página se podrá ver cómo funciona un modelo CONVOLUCIONAL en la predicción del mercado de valores"""
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

model = tf.keras.models.load_model('modelos/modelo_CONV.h5')

#calcular las predicciones(ahora para toda la variable y): 
#preparar los datos
# ajustar dimensiones de los datos de prueba
data = X  # Tu secuencia de datos
target =  y # La secuencia para comparar

# Crear el generador de series de tiempo
length = 10  # Tamaño de cada secuencia
batch_size = 32  # Tamaño del lote

generator_final = TimeseriesGenerator(data, np.zeros(len(data)), length=length, batch_size=batch_size)

predicciones = model.predict(generator_final) #predicciones

predicciones_array = np.squeeze(predicciones) #hacerlas un aaray de una dimension

umbral = 0.5
predicciones_binarias = (predicciones_array >= umbral).astype(int) #pasar segun la probabilidad a 1 o 0

datos_perdidos = len(y) - len(predicciones_binarias) #calcular los datos que no se usaron

df = df[datos_perdidos:] #reajustar el dataframe


#adjunto las predicciones realizadas al dataframe
df['Predicted_Signal'] = predicciones_binarias 

# Calcular retornos diarios (sin estrategia)
df['Return'] = df.Close.pct_change() #funcion para sacar el porcentaje de aumento o disminucion

# Calcular retornos con la estrategia
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)

# crear la columna con el retorno acumulado (sin estrategia)
df['Cum_Ret'] = df['Return'].cumsum()
st.write("Dataframe con retornos acumulativos")
df
# crear la columna con el retorno acumulado (con estrategia)
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