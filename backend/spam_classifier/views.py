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
        
        # Make prediction - get probabilities
        probabilities = model.predict_proba(email_vector)[0]
        classes = model.classes_
        
        # Map classes correctly: find which index corresponds to spam vs ham
        # Most models use 0=ham, 1=spam OR 'ham', 'spam'
        if len(classes) == 2:
            # Convert classes to strings for comparison
            class_0 = str(classes[0]).lower()
            class_1 = str(classes[1]).lower()
            
            # Determine which index is spam
            if class_1 in ['1', 'spam']:
                spam_idx = 1
                ham_idx = 0
            elif class_0 in ['1', 'spam']:
                spam_idx = 0
                ham_idx = 1
            else:
                # Default: assume higher index is spam
                spam_idx = 1
                ham_idx = 0
            
            spam_prob = float(probabilities[spam_idx])
            ham_prob = float(probabilities[ham_idx])
            
            # Determine prediction based on probability
            is_spam = spam_prob > 0.5
            prediction_label = 'spam' if is_spam else 'ham'
        else:
            # Fallback
            spam_prob = float(probabilities[-1])
            ham_prob = float(probabilities[0])
            prediction_label = 'spam' if spam_prob > 0.5 else 'ham'
        
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
            'prediction': prediction_label,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'important_words': top_words,
            'parsed_tokens_count': {
                'subject': len(parsed_email['subject']),
                'body': len(parsed_email['body'])
            },
            # Debug info to help diagnose issues
            'debug': {
                'model_classes': [str(c) for c in classes],
                'raw_probabilities': [float(p) for p in probabilities],
                'spam_index': spam_idx,
                'ham_index': ham_idx,
                'processed_text_length': len(processed_text.split()),
                'processed_text_preview': ' '.join(processed_text.split()[:50])
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
