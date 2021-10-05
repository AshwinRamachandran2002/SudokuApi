from django.contrib import admin
from django.urls import include,path
from django.views.generic import RedirectView
from django.views.generic.base import TemplateView
from digitrecognizer import views

urlpatterns= [
    path('admin/', admin.site.urls),
    path('', views.home,name='home'),
    path('main/',include(('main.urls','main'), namespace='main')),
]