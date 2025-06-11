from django.urls import path
from . import views

app_name = 'forecasting_app'  # Make sure this line is here

urlpatterns = [
    path('', views.index, name='index'),
    path('forecast/', views.forecast, name='forecast'),
]

