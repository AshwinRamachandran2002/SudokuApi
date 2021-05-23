from django.contrib import admin
from django.urls import path, include
from main.views import predict
urlpatterns = [
    path('', include('main.urls')),
    path('api/predict/', predict)
]