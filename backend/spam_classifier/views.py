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
        
        # Vectorize the email
        email_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(email_vector)[0]
        probabilities = model.predict_proba(email_vector)[0]
        
        # Get class labels (assuming binary classification: ham=0, spam=1)
        classes = model.classes_
        prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
        
        # Determine spam probability
        spam_prob = prob_dict.get('spam', prob_dict.get('1', 0.5))
        
        # Get feature importance (words with highest coefficients)
        feature_names = vectorizer.get_feature_names_out()
        email_features = email_vector.toarray()[0]
        
        # Get indices of non-zero features
        non_zero_indices = email_features.nonzero()[0]
        
        # Get model coefficients
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            
            # Calculate weighted importance for each word in the email
            word_importance = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                weight = float(coefficients[idx] * email_features[idx])
                word_importance.append({
                    'word': word,
                    'weight': weight
                })
            
            # Sort by absolute weight and take top 20
            word_importance.sort(key=lambda x: abs(x['weight']), reverse=True)
            top_words = word_importance[:20]
        else:
            top_words = []
        
        return JsonResponse({
            'prediction': str(prediction),
            'probabilities': prob_dict,
            'spam_probability': spam_prob,
            'ham_probability': 1 - spam_prob,
            'important_words': top_words,
            'parsed_tokens': {
                'subject': parsed_email['subject'][:10],  # First 10 tokens
                'body_preview': parsed_email['body'][:20]  # First 20 tokens
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON in request body'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'Server error: {str(e)}'
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
