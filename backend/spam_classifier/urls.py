from django.urls import path
from . import views

urlpatterns = [
    path('predict', views.predict, name='predict'),
    path('health', views.health, name='health'),
    path('info', views.model_info, name='model_info'),
]
