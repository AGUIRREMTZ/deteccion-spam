
        }, status=500)
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import joblib
import os
from .utils.email_parser import Parser

# Initialize parser
parser = Parser()

# Load models (will be loaded when first needed)
model = None
vectorizer = None


def load_models():
    """Load the spam detection model and vectorizer"""
    global model, vectorizer
    
    if model is None or vectorizer is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(base_dir, 'static')
        
        model_path = os.path.join(static_dir, 'modelo_spam.joblib')
        vectorizer_path = os.path.join(static_dir, 'vectorizador.joblib')
        
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
        except FileNotFoundError as e:
            raise Exception(f"Model files not found. Please ensure modelo_spam.joblib and vectorizador.joblib are in the static/ folder. Error: {str(e)}")


@csrf_exempt
@require_http_methods(["POST"])
def predict(request):
    """
    Endpoint to predict if an email is spam or ham
    """
    try:
        # Load models if not already loaded
        load_models()
        
        # Parse request body
        data = json.loads(request.body)
        email_content = data.get('email_content', '')
        
        if not email_content:
            return JsonResponse({
                'error': 'No email content provided'
            }, status=400)
        
        # Parse email using the same Parser class from training
        parsed_email = parser.parse_text(email_content)
        
        # Convert tokens to text for vectorization
        processed_text = parser.get_text_from_tokens(parsed_email)
        
        if not processed_text.strip():
            return JsonResponse({
                'prediction': 'ham',
                'spam_probability': 0.0,
                'ham_probability': 1.0,
                'important_words': [],
                'parsed_tokens_count': {
                    'subject': len(parsed_email['subject']),
                    'body': len(parsed_email['body'])
                },
                'warning': 'No processable text found in email'
            })
        
        # Vectorize the email
        email_vector = vectorizer.transform([processed_text])
        
        probabilities = model.predict_proba(email_vector)[0]
        prediction_class = model.predict(email_vector)[0]
        classes = model.classes_
        
        # Map classes to indices
        class_to_idx = {str(cls): idx for idx, cls in enumerate(classes)}
        
        # Determine spam and ham indices
        spam_idx = None
        ham_idx = None
        
        for idx, cls in enumerate(classes):
            cls_str = str(cls).lower()
            if cls_str in ['spam', '1', 'true']:
                spam_idx = idx
            elif cls_str in ['ham', 'normal', '0', 'false']:
                ham_idx = idx
        
        # Fallback: if not found, assume binary classification with spam as 1
        if spam_idx is None:
            if len(classes) == 2:
                spam_idx = 1
                ham_idx = 0
            else:
                spam_idx = 0
                ham_idx = 1 if len(classes) > 1 else 0
        
        if ham_idx is None:
            ham_idx = 1 - spam_idx if len(classes) == 2 else 0
        
        spam_prob = float(probabilities[spam_idx])
        ham_prob = float(probabilities[ham_idx]) if ham_idx < len(probabilities) else 1.0 - spam_prob
        
        # Determine final prediction
        is_spam = spam_prob > 0.5
        prediction_label = 'spam' if is_spam else 'ham'
        
        # Get feature importance (words with highest coefficients)
        feature_names = vectorizer.get_feature_names_out()
        email_features = email_vector.toarray()[0]
        
        # Get indices of non-zero features
        non_zero_indices = email_features.nonzero()[0]
        
        word_importance = []
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            # Calculate weighted importance for each word in the email
            for idx in non_zero_indices:
                if idx < len(feature_names):
                    word = feature_names[idx]
                    # Weight combines the coefficient and the term frequency
                    coef = float(coefficients[idx]) if idx < len(coefficients) else 0
                    tf = float(email_features[idx])
                    weight = coef * tf
                    
                    word_importance.append({
                        'word': word,
                        'weight': weight,
                        'coefficient': coef,
                        'frequency': tf
                    })
            
            # Sort by absolute weight and take top 20
            word_importance.sort(key=lambda x: abs(x['weight']), reverse=True)
            top_words = word_importance[:20]
        else:
            top_words = []
        
        return JsonResponse({
            'prediction': prediction_label,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'important_words': top_words,
            'parsed_tokens_count': {
                'subject': len(parsed_email['subject']),
                'body': len(parsed_email['body'])
            },
            'model_info': {
                'model_type': type(model).__name__,
                'classes': [str(c) for c in classes],
                'n_features': len(feature_names),
                'has_coefficients': hasattr(model, 'coef_')
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        import traceback
        return JsonResponse({
            'error': f'Server error: {str(e)}',
            'traceback': traceback.format_exc()
        }, status=500)


@require_http_methods(["GET"])
def health(request):
    """Health check endpoint"""
    try:
        load_models()
        return JsonResponse({
            'status': 'healthy',
            'models_loaded': True
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def model_info(request):
    """Get information about the loaded model"""
    try:
        load_models()
        
        info = {
            'model_type': type(model).__name__,
            'vectorizer_type': type(vectorizer).__name__,
            'vocabulary_size': len(vectorizer.get_feature_names_out()),
            'classes': list(model.classes_) if hasattr(model, 'classes_') else []
        }
        
        return JsonResponse(info)
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def model_metrics(request):
    """
    Get pre-calculated model performance metrics
    These metrics should be calculated once on a test set and stored
    """
    try:
        load_models()
        
        # In production, these would be calculated on a validation set and cached
        # For now, returning placeholder structure that matches expected format
        metrics = {
            'confusion_matrix': {
                'true_negative': 0,  # Ham correctly classified as Ham
                'false_positive': 0,  # Ham incorrectly classified as Spam
                'false_negative': 0,  # Spam incorrectly classified as Ham
                'true_positive': 0,   # Spam correctly classified as Spam
            },
            'performance_metrics': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'specificity': 0.0,
            },
            'note': 'Model metrics should be calculated on a test dataset. Upload modelo_metricas.json to static/ folder with actual metrics.'
        }
        
        # Try to load pre-calculated metrics from file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        static_dir = os.path.join(base_dir, 'static')
        metrics_path = os.path.join(static_dir, 'modelo_metricas.json')
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
        
        return JsonResponse(metrics)
        
    except Exception as e:
        return JsonResponse({
            'error': str(e)
        }, status=500)
