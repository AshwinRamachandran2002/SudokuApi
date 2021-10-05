from . import views
from django.conf.urls import url
from django.urls import path,include

urlpatterns = [
    url('solve',views.solve,name='solve'),
    url('retrain',views.retrain,name='retrain'),
]