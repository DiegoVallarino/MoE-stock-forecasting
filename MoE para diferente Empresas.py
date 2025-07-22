## Paper MoE para evaluacion de Acciones de Diferentes empresas

# Paso 1: Instalar dependencias
!pip install yfinance tensorflow scikit-learn --quiet

# Paso 2: Importar librerías
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Paso 3: Descargar datos de 30 empresas (diversas industrias)
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'AMD', 'NFLX', 'INTC',
    'KO', 'JNJ', 'PG', 'PEP', 'MCD', 'WMT', 'T', 'VZ', 'XOM', 'CVX',
    'BA', 'GE', 'CAT', 'NKE', 'DIS', 'ADBE', 'CRM', 'QCOM', 'CSCO', 'ORCL'
]

start_date = "2015-01-01"
end_date = "2024-12-31"

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

# Paso 4: Calcular retornos logarítmicos y volatilidad rolling
log_returns = np.log(data / data.shift(1))
vol21 = log_returns.rolling(window=21).std()

# Paso 5: Clasificar tickers por volatilidad promedio
vol_scores = vol21.mean()
threshold = vol_scores.median()
classification = {ticker: 'volatile' if vol_scores[ticker] > threshold else 'stable' for ticker in tickers}

# Paso 6: Construir dataset secuencial (20 días de input → retorno siguiente día)
window_size = 20
X, y, ticker_labels = [], [], []

for ticker in tickers:
    series = log_returns[ticker].dropna().values
    for i in range(window_size, len(series)-1):
        X.append(series[i-window_size:i])
        y.append(series[i+1])
        ticker_labels.append(ticker)

X = np.array(X)
y = np.array(y)
ticker_labels = np.array(ticker_labels)

# Paso 7: Separar por grupo
tickers_volatile = [t for t in tickers if classification[t] == 'volatile']
tickers_stable = [t for t in tickers if classification[t] == 'stable']

X_volatile = X[np.isin(ticker_labels, tickers_volatile)]
y_volatile = y[np.isin(ticker_labels, tickers_volatile)]

X_stable = X[np.isin(ticker_labels, tickers_stable)]
y_stable = y[np.isin(ticker_labels, tickers_stable)]

# Paso 8: Entrenar LSTM para volátiles
scaler_rnn = StandardScaler()
X_vol_scaled = scaler_rnn.fit_transform(X_volatile.reshape(X_volatile.shape[0], -1)).reshape(X_volatile.shape[0], window_size, 1)

rnn_model = Sequential()
rnn_model.add(LSTM(20, input_shape=(window_size, 1)))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_vol_scaled, y_volatile, epochs=10, batch_size=32, verbose=1)

# Paso 9: Entrenar regresión lineal para acciones estables
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Escalado para modelo lineal
scaler_lin = StandardScaler()
X_stable_scaled = scaler_lin.fit_transform(X_stable)

# Entrenar modelo lineal
lin_model = LinearRegression()
lin_model.fit(X_stable_scaled, y_stable)

# Paso 10 (optimizado): Predicción combinada vectorizada (MoE)
# Clasificación
tickers_volatile = [t for t in tickers if classification[t] == 'volatile']
tickers_stable = [t for t in tickers if classification[t] == 'stable']
window_size = X.shape[1]

# Volátiles → usar RNN + combinar con OLS
mask_vol = np.isin(ticker_labels, tickers_volatile)
X_vol_all = scaler_rnn.transform(X[mask_vol].reshape(-1, window_size)).reshape(-1, window_size, 1)
y_rnn_vol = rnn_model.predict(X_vol_all, verbose=0).flatten()
y_lin_vol = lin_model.predict(scaler_lin.transform(X[mask_vol]))
y_pred_vol = 0.8 * y_rnn_vol + 0.2 * y_lin_vol
y_true_vol = y[mask_vol]
group_vol = ["volatile"] * len(y_pred_vol)

# Estables → usar OLS + suavizar con RNN
mask_stab = np.isin(ticker_labels, tickers_stable)
X_stab_all = scaler_lin.transform(X[mask_stab])
y_lin_stab = lin_model.predict(X_stab_all)
X_stab_rnn = scaler_rnn.transform(X[mask_stab].reshape(-1, window_size)).reshape(-1, window_size, 1)
y_rnn_stab = rnn_model.predict(X_stab_rnn, verbose=0).flatten()
y_pred_stab = 0.2 * y_rnn_stab + 0.8 * y_lin_stab
y_true_stab = y[mask_stab]
group_stab = ["stable"] * len(y_pred_stab)

# Combinar todo
results_df = pd.DataFrame({
    "y_true": np.concatenate([y_true_vol, y_true_stab]),
    "y_pred": np.concatenate([y_pred_vol, y_pred_stab]),
    "group": group_vol + group_stab
})

# Paso 11: Evaluación de desempeño
group_metrics = results_df.groupby("group").apply(
    lambda df: pd.Series({
        "MSE": mean_squared_error(df["y_true"], df["y_pred"]),
        "MAE": mean_absolute_error(df["y_true"], df["y_pred"])
    })
)

# Métricas globales
group_metrics.loc["global"] = {
    "MSE": mean_squared_error(results_df["y_true"], results_df["y_pred"]),
    "MAE": mean_absolute_error(results_df["y_true"], results_df["y_pred"])
}

print("\n📊 Desempeño por grupo (MoE):")
print(group_metrics.round(6))

# Paso 12: Visualización del error por grupo
plt.figure(figsize=(10, 5))
plt.hist(
    results_df[results_df["group"] == "volatile"]["y_true"] - results_df[results_df["group"] == "volatile"]["y_pred"], 
    bins=50, alpha=0.5, label="Volatile Errors"
)
plt.hist(
    results_df[results_df["group"] == "stable"]["y_true"] - results_df[results_df["group"] == "stable"]["y_pred"], 
    bins=50, alpha=0.5, label="Stable Errors"
)
plt.axvline(0, color='black', linestyle='--')
plt.title("Distribución del error de predicción por grupo (MoE)")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.show()
