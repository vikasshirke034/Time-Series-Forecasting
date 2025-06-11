from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('forecasting_app.urls', namespace='forecasting_app')),  # Added namespace
]
