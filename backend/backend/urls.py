from django.urls import path, include

urlpatterns = [
    path('api/', include('spam_classifier.urls')),
]
