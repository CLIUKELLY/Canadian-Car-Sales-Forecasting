# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from neuralprophet import NeuralProphet
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import os
import torch

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Use an absolute path to the CSV file
file_path = os.path.join(os.path.dirname(__file__), 'New motor vehicle sales data.csv')

# Load and preprocess the data
data = pd.read_csv(file_path, parse_dates=True)
data['REF_DATE'] = pd.to_datetime(data['REF_DATE'])
data = data[data['GEO'] == 'Canada']  # Use national-level data
data = data[(data['Sales'] == 'Units') & (data['Seasonal adjustment'] == 'Unadjusted') &
            (data['Vehicle type'] == 'Total, new motor vehicles') &
            (data['Origin of manufacture'] == 'Total, country of manufacture')]

# Prepare time series
ts_data = data[['REF_DATE', 'VALUE']].set_index('REF_DATE')
ts_data = ts_data.sort_index()

# Last 72 months for analysis
ts_data_last_72_mths = ts_data.tail(72)

# Split train and test sets
test_period = 36
test_set = ts_data_last_72_mths.tail(test_period)
train_set = ts_data_last_72_mths.iloc[:-test_period]

print(f"Train set shape: {train_set.shape}")
print(f"Test set shape: {test_set.shape}")

# ------------------------------
# ARIMA Model
# ------------------------------
arima_model = ARIMA(train_set['VALUE'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
arima_result = arima_model.fit()

# Forecast
forecast_arima = arima_result.forecast(steps=test_period)
forecast_arima_df = pd.DataFrame({'Forecast': forecast_arima}, index=test_set.index)

# Evaluation
mae_arima = mean_absolute_error(test_set['VALUE'], forecast_arima)
rmse_arima = np.sqrt(mean_squared_error(test_set['VALUE'], forecast_arima))
mape_arima = np.mean(np.abs((test_set['VALUE'] - forecast_arima) / test_set['VALUE'])) * 100

print(f"ARIMA - MAE: {mae_arima:.2f}, RMSE: {rmse_arima:.2f}, MAPE: {mape_arima:.2f}%")

# ------------------------------
# NeuralProphet Model
# ------------------------------

# Allow NeuralProphet's ConfigSeasonality to be safely loaded
# torch.serialization.add_safe_globals([NeuralProphet.configure.ConfigSeasonality])
# torch.serialization.safe_globals([neuralprophet.configure.ConfigSeasonality])


# Prepare data for NeuralProphet
train_np = train_set.reset_index().rename(columns={'REF_DATE': 'ds', 'VALUE': 'y'})
test_np = test_set.reset_index().rename(columns={'REF_DATE': 'ds', 'VALUE': 'y'})

np_model = NeuralProphet(
    growth='linear',
    changepoints_range=0.8,
    n_changepoints=10,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode='additive',
    epochs=100,
    learning_rate=0.01
)
np_model.fit(train_np, freq='MS')

# Forecast
future_np = np_model.make_future_dataframe(train_np, periods=test_period)
forecast_np = np_model.predict(future_np)
forecast_neural_prophet = forecast_np.tail(test_period)['yhat1'].values

# Evaluation
mae_np = mean_absolute_error(test_set['VALUE'], forecast_neural_prophet)
rmse_np = np.sqrt(mean_squared_error(test_set['VALUE'], forecast_neural_prophet))
mape_np = np.mean(np.abs((test_set['VALUE'] - forecast_neural_prophet) / test_set['VALUE'])) * 100

print(f"NeuralProphet - MAE: {mae_np:.2f}, RMSE: {rmse_np:.2f}, MAPE: {mape_np:.2f}%")

# ------------------------------
# Prophet Model
# ------------------------------
train_prophet = train_set.reset_index().rename(columns={'REF_DATE': 'ds', 'VALUE': 'y'})

prophet_model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
prophet_model.fit(train_prophet)

# Forecast
future_prophet = prophet_model.make_future_dataframe(periods=test_period, freq='MS')
forecast_prophet = prophet_model.predict(future_prophet)
forecast_prophet_values = forecast_prophet.tail(test_period)['yhat'].values

# Evaluation
mae_prophet = mean_absolute_error(test_set['VALUE'], forecast_prophet_values)
rmse_prophet = np.sqrt(mean_squared_error(test_set['VALUE'], forecast_prophet_values))
mape_prophet = np.mean(np.abs((test_set['VALUE'] - forecast_prophet_values) / test_set['VALUE'])) * 100

print(f"Prophet - MAE: {mae_prophet:.2f}, RMSE: {rmse_prophet:.2f}, MAPE: {mape_prophet:.2f}%")

# ------------------------------
# LSTM Model
# ------------------------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_set.values.reshape(-1, 1))
test_scaled = scaler.transform(test_set.values.reshape(-1, 1))

# Create sequences
LOOKBACK_WINDOW = 12
X_train, y_train = [], []
for i in range(LOOKBACK_WINDOW, len(train_scaled)):
    X_train.append(train_scaled[i-LOOKBACK_WINDOW:i])
    y_train.append(train_scaled[i])
X_train, y_train = np.array(X_train), np.array(y_train)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(LOOKBACK_WINDOW, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Forecast
last_sequence = train_scaled[-LOOKBACK_WINDOW:]
predictions = []
for i in range(test_period):
    pred = lstm_model.predict(last_sequence.reshape(1, LOOKBACK_WINDOW, 1), verbose=0)
    predictions.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0])
predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# Evaluation
mae_lstm = mean_absolute_error(test_set['VALUE'], predictions_rescaled)
rmse_lstm = np.sqrt(mean_squared_error(test_set['VALUE'], predictions_rescaled))
mape_lstm = np.mean(np.abs((test_set['VALUE'] - predictions_rescaled) / test_set['VALUE'])) * 100

print(f"LSTM - MAE: {mae_lstm:.2f}, RMSE: {rmse_lstm:.2f}, MAPE: {mape_lstm:.2f}%")

# ------------------------------
# Side-by-Side Comparison
# ------------------------------
results = pd.DataFrame({
    'Model': ['ARIMA', 'NeuralProphet', 'Prophet', 'LSTM'],
    'MAE': [mae_arima, mae_np, mae_prophet, mae_lstm],
    'RMSE': [rmse_arima, rmse_np, rmse_prophet, rmse_lstm],
    'MAPE': [mape_arima, mape_np, mape_prophet, mape_lstm]
})

print("\nModel Performance Comparison:")
print(results)

# Plot actual vs predicted for all models
plt.figure(figsize=(20, 10))

# Actual values
plt.plot(ts_data_last_72_mths.index, ts_data_last_72_mths['VALUE'], label='Actual', color='black', linewidth=2)

# ARIMA
plt.plot(forecast_arima_df.index, forecast_arima_df['Forecast'], label='ARIMA', linestyle='--')

# NeuralProphet
plt.plot(test_set.index, forecast_neural_prophet, label='NeuralProphet', linestyle='--')

# Prophet
plt.plot(test_set.index, forecast_prophet_values, label='Prophet', linestyle='--')

# LSTM
plt.plot(test_set.index, predictions_rescaled, label='LSTM', linestyle='--')

plt.axvline(x=train_set.index[-1], color='red', linestyle='--', label='Train/Test Split')
plt.title('Model Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
