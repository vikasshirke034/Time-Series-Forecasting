from django import forms

class ForecastingForm(forms.Form):
    DATA_TYPES = [
        ('stock', 'Stock Prices'),
        ('sales', 'Sales Data'),
        ('weather', 'Weather Data')
    ]
    
    MODELS = [
        ('all', 'All Models'),
        ('arima', 'ARIMA'),
        ('prophet', 'Prophet'),
        ('lstm', 'LSTM')
    ]
    
    data_type = forms.ChoiceField(
        choices=DATA_TYPES,
        label="Select Data Type",
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50'})
    )
    
    model_type = forms.ChoiceField(
        choices=MODELS,
        label="Select Model",
        widget=forms.Select(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50'})
    )
    
    forecast_periods = forms.IntegerField(
        min_value=1,
        max_value=90,
        initial=30,
        label="Forecast Horizon (days)",
        widget=forms.NumberInput(attrs={'class': 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50'})
    )