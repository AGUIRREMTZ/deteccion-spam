from django.urls import path, include
from django.http import JsonResponse

def home(request):
    """Root endpoint with API documentation"""
    return JsonResponse({
        'message': 'Spam Detection API',
        'version': '1.0',
        'endpoints': {
            '/': 'API documentation (this page)',
            '/api/health/': 'Health check endpoint',
            '/api/predict/': 'POST - Predict if email is spam',
            '/api/model-info/': 'GET - Get model information'
        },
        'status': 'running'
    })

urlpatterns = [
    path('', home, name='home'),
    path('api/', include('spam_classifier.urls')),
]
