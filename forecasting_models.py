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
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    def __init__(self, file_path, test_period=36, lookback_window=12):
        """
        Initialize the forecaster with data and parameters
        
        Args:
            file_path (str): Path to the CSV file
            test_period (int): Number of periods for testing
            lookback_window (int): Window size for LSTM
        """
        self.file_path = file_path
        self.test_period = test_period
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler()
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Initialize data containers
        self.data = None
        self.ts_data = None
        self.train_set = None
        self.test_set = None
        self.results = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the time series data"""
        try:
            # Load data
            self.data = pd.read_csv(self.file_path, parse_dates=True)
            self.data['REF_DATE'] = pd.to_datetime(self.data['REF_DATE'])
            
            # Filter data
            self.data = self.data[self.data['GEO'] == 'Canada']
            self.data = self.data[
                (self.data['Sales'] == 'Units') & 
                (self.data['Seasonal adjustment'] == 'Unadjusted') &
                (self.data['Vehicle type'] == 'Total, new motor vehicles') &
                (self.data['Origin of manufacture'] == 'Total, country of manufacture')
            ]
            
            # Prepare time series
            self.ts_data = self.data[['REF_DATE', 'VALUE']].set_index('REF_DATE')
            self.ts_data = self.ts_data.sort_index()
            
            # Last 72 months for analysis
            ts_data_last_72_mths = self.ts_data.tail(72)
            
            # Split train and test sets
            self.test_set = ts_data_last_72_mths.tail(self.test_period)
            self.train_set = ts_data_last_72_mths.iloc[:-self.test_period]
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_arima(self):
        """Train ARIMA model and make predictions"""
        try:
            arima_model = ARIMA(self.train_set['VALUE'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            arima_result = arima_model.fit()
            
            # Forecast
            forecast_arima = arima_result.forecast(steps=self.test_period)
            
            # Evaluation
            mae = mean_absolute_error(self.test_set['VALUE'], forecast_arima)
            rmse = np.sqrt(mean_squared_error(self.test_set['VALUE'], forecast_arima))
            mape = np.mean(np.abs((self.test_set['VALUE'] - forecast_arima) / self.test_set['VALUE'])) * 100
            
            self.results['ARIMA'] = {
                'predictions': forecast_arima,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            return True
            
        except Exception as e:
            print(f"Error training ARIMA: {e}")
            return False
    
    def train_neural_prophet(self):
        """Train NeuralProphet model and make predictions"""
        try:
            # Prepare data
            train_np = self.train_set.reset_index().rename(columns={'REF_DATE': 'ds', 'VALUE': 'y'})
            
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
            future_np = np_model.make_future_dataframe(train_np, periods=self.test_period)
            forecast_np = np_model.predict(future_np)
            predictions = forecast_np.tail(self.test_period)['yhat1'].values
            
            # Evaluation
            mae = mean_absolute_error(self.test_set['VALUE'], predictions)
            rmse = np.sqrt(mean_squared_error(self.test_set['VALUE'], predictions))
            mape = np.mean(np.abs((self.test_set['VALUE'] - predictions) / self.test_set['VALUE'])) * 100
            
            self.results['NeuralProphet'] = {
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            return True
            
        except Exception as e:
            print(f"Error training NeuralProphet: {e}")
            return False
    
    def train_prophet(self):
        """Train Prophet model and make predictions"""
        try:
            train_prophet = self.train_set.reset_index().rename(columns={'REF_DATE': 'ds', 'VALUE': 'y'})
            
            prophet_model = Prophet(yearly_seasonality=True, seasonality_mode='additive')
            prophet_model.fit(train_prophet)
            
            # Forecast
            future_prophet = prophet_model.make_future_dataframe(periods=self.test_period, freq='MS')
            forecast_prophet = prophet_model.predict(future_prophet)
            predictions = forecast_prophet.tail(self.test_period)['yhat'].values
            
            # Evaluation
            mae = mean_absolute_error(self.test_set['VALUE'], predictions)
            rmse = np.sqrt(mean_squared_error(self.test_set['VALUE'], predictions))
            mape = np.mean(np.abs((self.test_set['VALUE'] - predictions) / self.test_set['VALUE'])) * 100
            
            self.results['Prophet'] = {
                'predictions': predictions,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            return True
            
        except Exception as e:
            print(f"Error training Prophet: {e}")
            return False
    
    def train_lstm(self):
        """Train LSTM model and make predictions"""
        try:
            # Scale data
            train_scaled = self.scaler.fit_transform(self.train_set.values.reshape(-1, 1))
            
            # Create sequences
            X_train, y_train = [], []
            for i in range(self.lookback_window, len(train_scaled)):
                X_train.append(train_scaled[i-self.lookback_window:i])
                y_train.append(train_scaled[i])
            X_train, y_train = np.array(X_train), np.array(y_train)
            
            # Build LSTM model
            lstm_model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(self.lookback_window, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            lstm_model.compile(optimizer='adam', loss='mse')
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Forecast
            last_sequence = train_scaled[-self.lookback_window:]
            predictions = []
            for i in range(self.test_period):
                pred = lstm_model.predict(last_sequence.reshape(1, self.lookback_window, 1), verbose=0)
                predictions.append(pred[0, 0])
                last_sequence = np.append(last_sequence[1:], pred[0, 0])
            predictions_rescaled = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
            
            # Evaluation
            mae = mean_absolute_error(self.test_set['VALUE'], predictions_rescaled)
            rmse = np.sqrt(mean_squared_error(self.test_set['VALUE'], predictions_rescaled))
            mape = np.mean(np.abs((self.test_set['VALUE'] - predictions_rescaled) / self.test_set['VALUE'])) * 100
            
            self.results['LSTM'] = {
                'predictions': predictions_rescaled,
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
            
            return True
            
        except Exception as e:
            print(f"Error training LSTM: {e}")
            return False
    
    def train_all_models(self):
        """Train all models"""
        if not self.load_and_preprocess_data():
            return False
        
        models_success = {
            'ARIMA': self.train_arima(),
            'NeuralProphet': self.train_neural_prophet(),
            'Prophet': self.train_prophet(),
            'LSTM': self.train_lstm()
        }
        
        return models_success
    
    def get_results_dataframe(self):
        """Get results as a pandas DataFrame"""
        if not self.results:
            return None
        
        results_data = []
        for model_name, metrics in self.results.items():
            results_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'MAPE': metrics['mape']
            })
        
        return pd.DataFrame(results_data)
    
    def create_comparison_plot(self):
        """Create comparison plot of all models"""
        if not self.results:
            return None
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot actual values
        full_data = pd.concat([self.train_set, self.test_set])
        ax.plot(full_data.index, full_data['VALUE'], label='Actual', color='black', linewidth=2)
        
        # Plot predictions for each model
        colors = ['red', 'blue', 'green', 'orange']
        for i, (model_name, model_results) in enumerate(self.results.items()):
            ax.plot(self.test_set.index, model_results['predictions'], 
                   label=f'{model_name}', linestyle='--', color=colors[i % len(colors)])
        
        # Add vertical line for train/test split
        ax.axvline(x=self.train_set.index[-1], color='red', linestyle=':', alpha=0.7, label='Train/Test Split')
        
        ax.set_title('Model Forecast Comparison', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Sales', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
