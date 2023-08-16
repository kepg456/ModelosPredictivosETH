# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# For data manipulation
import yfinance as yf
import pandas as pd
import numpy as np
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")
import streamlit as st

warnings.filterwarnings("ignore")

st.markdown("# Modelo Predictivo - Suport Vector Machine(SVM)")
st.sidebar.header("SVM")
st.write(
    """En esta página se podrá ver cómo funciona un modelo Suport Vector Machine en la predicción del mercado de valores"""
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

#dividir datos en entrenamiento y prueba
split_percentage = 0.8
split = int(split_percentage*len(df))

# Train data set
X_train = X[:split]
y_train = y[:split]

# Test data set
X_test = X[split:]
y_test = y[split:]

# Support vector classifier
cls = SVC().fit(X_train, y_train)

df['Predicted_Signal'] = cls.predict(X)

# Calculate daily returns
df['Return'] = df.Close.pct_change()

# Calculate strategy returns
df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)


# Calculate Cumulutive returns
df['Cum_Ret'] = df['Return'].cumsum()
st.write("Dataframe con retornos  acumulativos")
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