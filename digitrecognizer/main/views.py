from django.shortcuts import render

def index(request):
    return render(request, 'index.html', {})
# urls.py
from django.urls import path
from .views import index
urlpatterns = [
    path('', index, name="index")
]