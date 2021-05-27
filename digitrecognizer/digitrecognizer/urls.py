from django.contrib import admin
from django.urls import include,path
#from main.views import predict
urlpatterns= [
    path('admin/', admin.site.urls),
    path('', include('main.views')),
]