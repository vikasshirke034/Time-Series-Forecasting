# Time Series Forecasting Web Application

A Django-based web application that implements multiple time series forecasting models (ARIMA, Prophet, and LSTM) to predict future values for different types of data (stock prices, sales data, and weather data).

## Features

- Multiple forecasting models:
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Prophet (Facebook's time series forecasting tool)
  - LSTM (Long Short-Term Memory neural network)
- Support for different data types:
  - Stock price data
  - Sales data
  - Weather data
- Interactive visualizations using Plotly
- Model performance metrics and comparisons
- Configurable forecast horizon
- Advanced data preprocessing and normalization

## Dataset Description

The application uses three types of time series data:

1. **Stock Price Data**:
   - Daily stock price values
   - Features: trend, seasonality, volatility
   - Includes technical indicators and market patterns
   - Located in: `data/stock_data.csv`

2. **Sales Data**:
   - Daily sales records
   - Features: weekly patterns, seasonal trends, special events
   - Includes business day effects and holiday impacts
   - Located in: `data/sales_data.csv`

3. **Weather Data**:
   - Daily temperature readings
   - Features: seasonal patterns, daily variations
   - Includes yearly cycles and weather patterns
   - Located in: `data/weather_data.csv`

## Installation

1. Clone the repository:
```bash
git clone 
cd forecasting_project
```

2. Create and activate a virtual environment:
```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up the database:
```bash
python manage.py migrate
```



## Running the Application

1. Start the Django development server:
```bash
python manage.py runserver
```

2. Open a web browser and navigate to:
```
http://127.0.0.1:8000/
```

## Forecasting Models

### 1. ARIMA Model
- Automatically finds optimal parameters (p,d,q) using AIC
- Handles trend and seasonality
- Provides confidence intervals
- Best for: Linear time series with clear patterns

### 2. Prophet Model
- Handles missing data automatically
- Captures multiple seasonalities
- Adjusts for holidays and events
- Best for: Data with strong seasonal patterns

### 3. LSTM Model
- Deep learning approach for complex patterns
- Uses multiple features for prediction
- Handles non-linear relationships
- Best for: Complex time series with long-term dependencies

## Project Structure
```
forecasting_project/
├── data/
│   ├── stock_data.csv
│   ├── sales_data.csv
│   └── weather_data.csv
├── forecasting_app/
│   ├── templates/
│   ├── static/
│   ├── utils.py
│   ├── views.py
│   └── models.py
├── manage.py
└── requirements.txt
```

## Implementation Challenges and Solutions

1. **Data Preprocessing**
   - Challenge: Handling missing values and outliers
   - Solution: Implemented robust preprocessing pipeline with multiple techniques
   - Impact: Improved model accuracy and reliability

2. **Model Optimization**
   - Challenge: Finding optimal parameters for each model
   - Solution: Implemented automated parameter selection and cross-validation
   - Impact: Better model performance across different data types

3. **Scalability**
   - Challenge: Processing large datasets efficiently
   - Solution: Implemented batch processing and optimized data handling
   - Impact: Improved application performance

4. **Model Comparison**
   - Challenge: Fairly comparing different types of models
   - Solution: Standardized evaluation metrics and visualization
   - Impact: Better model selection for different scenarios

5. **Visualization**
   - Challenge: Displaying complex time series data effectively
   - Solution: Interactive Plotly charts with multiple views
   - Impact: Better user understanding of forecasts

## Performance Metrics

The application evaluates models using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R-squared (R²) Score
- Model-specific metrics

## Future Improvements

1. Support for more data types
2. Additional forecasting models
3. Advanced feature engineering
4. Model ensemble capabilities
5. API endpoints for predictions

## Dependencies

- Django
- NumPy
- Pandas
- Scikit-learn
- Statsmodels
- Prophet
- TensorFlow
- Plotly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



## Acknowledgments

- Prophet by Facebook Research
- Plotly for visualization
- TensorFlow team for LSTM implementation
