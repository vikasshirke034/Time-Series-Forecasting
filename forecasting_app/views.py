# views.py
from django.shortcuts import render, redirect
from .forms import ForecastingForm
from .utils import ARIMAModel, ProphetModel, LSTMModel, create_forecast_plot, evaluate_model
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def index(request):
    form = ForecastingForm()
    return render(request, 'forecasting_app/index.html', {'form': form})

def forecast(request):
    if request.method == 'POST':
        form = ForecastingForm(request.POST)
        if form.is_valid():
            try:
                # Log form data
                logger.info("Form data received: %s", form.cleaned_data)
                
                data_type = form.cleaned_data['data_type']
                model_type = form.cleaned_data['model_type']
                periods = form.cleaned_data['forecast_periods']
                #confidence = form.cleaned_data['confidence_interval']
                
                # Initialize models
                models = {
                    'ARIMA': ARIMAModel(data_type),
                    'Prophet': ProphetModel(data_type),
                    'LSTM': LSTMModel(data_type)
                }
                
                # Load data
                data = models['ARIMA'].load_data()
                logger.info(f"Data loaded successfully. Shape: {data.shape}")
                
                forecasts = []
                metrics = {}
                model_names = []
                
                if model_type == 'all':
                    selected_models = models.items()
                else:
                    selected_models = [(k, v) for k, v in models.items() 
                                     if k.lower() == model_type]
                
                logger.info(f"Selected models: {[name for name, _ in selected_models]}")
                
                for name, model in selected_models:
                    logger.info(f"Training model: {name}")
                    # Train model
                    model.train()
                    
                    # Make predictions
                    logger.info(f"Making predictions with {name}")
                    forecast, conf_int = model.predict(periods)
                    forecasts.append((forecast, conf_int))
                    model_names.append(name)
                    
                    # Calculate metrics
                    metrics[name] = evaluate_model(
                        data['value'].iloc[-periods:],
                        forecast[:periods]
                    )
                    logger.info(f"Metrics for {name}: {metrics[name]}")
                
                # Create visualization
                plot = create_forecast_plot(
                    data, forecasts, model_names,
                    f'{data_type.title()} Forecasting Results'
                )
                
                # Create summary statistics
                summary_stats = {
                    'Total Observations': len(data),
                    'Forecast Horizon': periods,
                    'Data Range': f"{data['date'].min().strftime('%Y-%m-%d')} to {data['date'].max().strftime('%Y-%m-%d')}",
                    'Average Value': f"{data['value'].mean():.2f}",
                    'Standard Deviation': f"{data['value'].std():.2f}"
                }
                logger.info(f"Summary statistics: {summary_stats}")

                # Create comparison table data
                comparison_data = []
                for name, metric in metrics.items():
                    row = {'Model': name}
                    row.update({k: f"{v:.4f}" for k, v in metric.items()})
                    comparison_data.append(row)
                logger.info(f"Comparison data: {comparison_data}")

                # Create forecast table data
                forecast_table = pd.DataFrame({
                    'Date': pd.date_range(start=data['date'].iloc[-1], periods=periods+1)[1:],
                    'Actual': ['N/A'] * periods
                })
                
                for name, (forecast, _) in zip(model_names, forecasts):
                    forecast_table[f'{name} Forecast'] = [f"{v:.2f}" for v in forecast]
                logger.info(f"Forecast table head: \n{forecast_table.head()}")

                context = {
                    'form': form,
                    'plot_div': plot.to_html(full_html=False, include_plotlyjs=True),
                    'metrics': metrics,
                    'data_type': data_type,
                    'model_names': model_names,
                    'summary_stats': summary_stats,
                    'comparison_data': comparison_data,
                    'forecast_table': forecast_table.to_dict('records'),
                    'column_names': forecast_table.columns.tolist()
                }
                
                # Log important context values
                logger.info("Context prepared successfully")
                logger.info(f"Data type: {data_type}")
                logger.info(f"Model names: {model_names}")
                logger.info(f"Metrics: {metrics}")
                
                return render(request, 'forecasting_app/forecast.html', context)
                
            except Exception as e:
                logger.error(f"Error occurred: {str(e)}", exc_info=True)
                context = {
                    'form': form,
                    'error_message': f"An error occurred: {str(e)}"
                }
                return render(request, 'forecasting_app/index.html', context)
    
    return render(request, 'forecasting_app/index.html', {'form': ForecastingForm()})