import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import logging
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TimeSeriesModel:
    def __init__(self, data_type):
        self.data_type = data_type
        self.data = None
        self.original_data = None
        self.scaler = MinMaxScaler()
    
    def handle_missing_values(self, df):
        """Handle missing values in the data"""
        df = df.copy()
        
        # Log missing values
        missing_before = df['value'].isna().sum()
        logger.info(f"Missing values before handling: {missing_before}")
        
        # Forward fill with limit
        df['value'] = df['value'].fillna(method='ffill', limit=3)
        
        # Backward fill with limit
        df['value'] = df['value'].fillna(method='bfill', limit=3)
        
        # Interpolate remaining missing values
        df['value'] = df['value'].interpolate(method='linear')
        
        # If any remaining NaNs (at edges), fill with mean
        if df['value'].isna().any():
            df['value'] = df['value'].fillna(df['value'].mean())
        
        missing_after = df['value'].isna().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return df
    
    def load_data(self):
        """Load and preprocess data"""
        try:
            # Load data
            data_path = f'data/{self.data_type}_data.csv'
            self.original_data = pd.read_csv(data_path)
            self.original_data['date'] = pd.to_datetime(self.original_data['date'])
            
            # Handle missing values
            self.data = self.handle_missing_values(self.original_data)
            
            logger.info(f"Data range: {self.data['value'].min():.2f} to {self.data['value'].max():.2f}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

class ARIMAModel(TimeSeriesModel):
    def __init__(self, data_type):
        super().__init__(data_type)
        self.best_order = None
        
    def find_best_order(self, data):
        """Find best ARIMA parameters using AIC"""
        best_aic = float('inf')
        best_order = (1,1,1)
        
        # Grid search for best parameters
        p_values = range(0, 3)
        d_values = range(0, 2)
        q_values = range(0, 3)
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    try:
                        model = ARIMA(data, order=(p,d,q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p,d,q)
                    except:
                        continue
        
        logger.info(f"Best ARIMA order: {best_order}, AIC: {best_aic}")
        return best_order
    
    def train(self):
        if self.data is None:
            self.load_data()
        
        # Ensure no NaN values
        if np.isnan(self.data['value']).any():
            raise ValueError("Data contains NaN values before ARIMA training")
        
        # Find best parameters
        self.best_order = self.find_best_order(self.data['value'])
        
        # Train model with best parameters
        self.model = ARIMA(self.data['value'], order=self.best_order)
        self.model_fit = self.model.fit()
        
        logger.info(f"ARIMA model trained with order {self.best_order}")
        return self.model_fit
    
    def predict(self, periods):
        forecast = self.model_fit.forecast(steps=periods)
        conf_int = self.model_fit.get_forecast(steps=periods).conf_int()
        # Ensure no NaN values in predictions
        if np.isnan(forecast).any():
            forecast = np.nan_to_num(forecast, nan=self.data['value'].mean())
        return forecast, {
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1]
        }

class ProphetModel(TimeSeriesModel):
    def train(self):
        if self.data is None:
            self.load_data()
        
        # Create Prophet dataframe
        prophet_df = self.data.copy()
        prophet_df = prophet_df.rename(columns={'date': 'ds', 'value': 'y'})
        
        # Configure Prophet with optimized parameters
        self.model = Prophet(
            changepoint_prior_scale=0.05,  # Flexibility of trend changes
            seasonality_prior_scale=10,    # Flexibility of seasonality
            holidays_prior_scale=10,       # Flexibility of holiday effects
            daily_seasonality=False,       # Disable daily seasonality for daily data
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative',  # Better for data with increasing trends
            changepoint_range=0.9          # Consider 90% of data for changepoints
        )
        
        # Add custom seasonalities based on data type
        if self.data_type == 'sales':
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            self.model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=3
            )
        elif self.data_type == 'stock':
            self.model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=3
            )
            # Add weekly seasonality for trading patterns
            self.model.add_seasonality(
                name='weekly',
                period=5,  # 5-day trading week
                fourier_order=3
            )
            
        # Fit model
        self.model.fit(prophet_df)
        logger.info("Prophet model trained with optimized parameters")
        return self.model
    
    def predict(self, periods):
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        predictions = forecast.yhat.values[-periods:]
        
        # Handle any NaN in predictions
        if np.isnan(predictions).any():
            predictions = np.nan_to_num(predictions, nan=self.data['value'].mean())
            
        return predictions, {
            'lower': forecast.yhat_lower.values[-periods:],
            'upper': forecast.yhat_upper.values[-periods:]
        }

class LSTMModel(TimeSeriesModel):
    def __init__(self, data_type):
        super().__init__(data_type)
        self.sequence_length = 30
        self.model = None
        self.value_scaler = MinMaxScaler(feature_range=(-1, 1))  # For the main value
        self.feature_scalers = {}  # For additional features
        
    def create_features(self, data):
        """Create additional features for better prediction"""
        df = pd.DataFrame(data, columns=['value'])
        
        # Technical indicators
        df['ma7'] = df['value'].rolling(window=7).mean()
        df['ma14'] = df['value'].rolling(window=14).mean()
        df['std7'] = df['value'].rolling(window=7).std()
        df['roc'] = df['value'].pct_change(3)
        
        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    def prepare_data_for_lstm(self, sequence_length=None):
        """Prepare data with correct scaling"""
        if sequence_length is None:
            sequence_length = self.sequence_length
            
        if self.data is None:
            self.load_data()
        
        # Create and scale features
        df = self.create_features(self.data['value'].values)
        
        # Scale the main value
        scaled_value = self.value_scaler.fit_transform(df['value'].values.reshape(-1, 1))
        
        # Scale other features independently
        scaled_features = np.zeros((len(df), len(df.columns)))
        scaled_features[:, 0] = scaled_value.ravel()  # First column is the main value
        
        for i, column in enumerate(df.columns[1:], 1):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            self.feature_scalers[column] = scaler
            scaled_features[:, i] = scaler.fit_transform(df[column].values.reshape(-1, 1)).ravel()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:(i + sequence_length)])
            y.append(scaled_value[i + sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        return (X_train, y_train), (X_val, y_val)
    
    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, input_shape=input_shape, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model
    
    def train(self):
        """Train LSTM model"""
        try:
            (X_train, y_train), (X_val, y_val) = self.prepare_data_for_lstm()
            
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
            return self.model
            
        except Exception as e:
            logger.error(f"Error in LSTM training: {str(e)}")
            raise
    
    def predict(self, periods):
        """Generate predictions with correct scaling"""
        try:
            if self.model is None:
                raise ValueError("Model not trained. Call train() first.")
            
            # Get last sequence of actual values
            last_sequence = self.data['value'].values[-self.sequence_length:]
            
            # Create features for last sequence
            df_last = self.create_features(last_sequence)
            
            # Scale features
            scaled_features = np.zeros((self.sequence_length, len(df_last.columns)))
            scaled_features[:, 0] = self.value_scaler.transform(df_last['value'].values.reshape(-1, 1)).ravel()
            
            for i, column in enumerate(df_last.columns[1:], 1):
                scaler = self.feature_scalers[column]
                scaled_features[:, i] = scaler.transform(df_last[column].values.reshape(-1, 1)).ravel()
            
            # Generate predictions
            predictions = []
            current_sequence = scaled_features.copy()
            
            for _ in range(periods):
                # Make prediction
                current_input = current_sequence.reshape(1, self.sequence_length, -1)
                pred = self.model.predict(current_input, verbose=0)[0, 0]
                predictions.append(pred)
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=0)
                # Update features for next prediction
                current_sequence[-1] = np.array([pred] + [pred] * (current_sequence.shape[1] - 1))
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.value_scaler.inverse_transform(predictions)
            
            logger.info(f"Generated {periods} predictions. Range: {predictions.min():.2f} to {predictions.max():.2f}")
            return predictions.flatten(), None
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {str(e)}")
            mean_value = self.data['value'].mean()
            return np.array([mean_value] * periods), None

def evaluate_model(actual, predicted):
    """Safely evaluate model performance"""
    try:
        # Remove any NaN values
        mask = ~(np.isnan(actual) | np.isnan(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0 or len(predicted) == 0:
            logger.warning("No valid data points for evaluation after removing NaN values")
            return {
                'MAE': np.nan,
                'RMSE': np.nan,
                'R2': np.nan
            }
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return {
            'MAE': np.nan,
            'RMSE': np.nan,
            'R2': np.nan
        }

def create_forecast_plot(data, forecasts, model_names, title):
    """Create visualization of forecasts using original scale data"""
    fig = make_subplots(rows=2, cols=1,
                       subplot_titles=(
                           'Historical Data and Forecasts',
                           'Model Comparison - Last 30 Days'
                       ),
                       vertical_spacing=0.15)
    
    # Plot historical data
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['value'],
                  name='Historical Data',
                  line=dict(color='black')),
        row=1, col=1
    )
    
    colors = ['blue', 'red', 'green']
    
    # Plot forecasts
    for (forecast, conf_int), model_name, color in zip(forecasts, model_names, colors):
        future_dates = pd.date_range(start=data['date'].iloc[-1],
                                   periods=len(forecast)+1)[1:]
        
        fig.add_trace(
            go.Scatter(x=future_dates,
                      y=forecast,
                      name=f'{model_name} Forecast',
                      line=dict(color=color)),
            row=1, col=1
        )
        
        if conf_int is not None and isinstance(conf_int, dict):
            if 'upper' in conf_int and 'lower' in conf_int:
                fig.add_trace(
                    go.Scatter(x=future_dates,
                              y=conf_int['upper'],
                              fill=None,
                              mode='lines',
                              line=dict(color=color, width=0),
                              showlegend=False),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=future_dates,
                              y=conf_int['lower'],
                              fill='tonexty',
                              mode='lines',
                              line=dict(color=color, width=0),
                              name=f'{model_name} CI'),
                    row=1, col=1
                )
    
    # Plot comparison for last 30 days
    last_30_actual = data['value'].iloc[-30:].values
    x_range = list(range(30))
    
    fig.add_trace(
        go.Scatter(x=x_range,
                  y=last_30_actual,
                  name='Actual',
                  line=dict(color='black', dash='dash')),
        row=2, col=1
    )
    
    for (forecast, _), model_name, color in zip(forecasts, model_names, colors):
        fig.add_trace(
            go.Scatter(x=x_range,
                      y=forecast[:30],
                      name=f'{model_name}',
                      line=dict(color=color)),
            row=2, col=1
        )
    
    fig.update_layout(
        height=800,
        title_text=title,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    
    return fig