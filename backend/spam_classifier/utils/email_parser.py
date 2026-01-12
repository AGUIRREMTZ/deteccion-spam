from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import joblib
import os
import numpy as np
from .utils.email_parser import Parser

# Try both configurations to see which works better
parser_with_stemming = Parser(use_stemming=True, remove_stopwords=False)
parser_no_stemming = Parser(use_stemming=False, remove_stopwords=False)

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
        
        parsed_with_stem = parser_with_stemming.parse_text(email_content)
        parsed_no_stem = parser_no_stemming.parse_text(email_content)
        
        text_with_stem = parser_with_stemming.get_text_from_tokens(parsed_with_stem)
        text_no_stem = parser_no_stemming.get_text_from_tokens(parsed_no_stem)
        
        # Vectorize both versions
        vector_with_stem = vectorizer.transform([text_with_stem])
        vector_no_stem = vectorizer.transform([text_no_stem])
        
        # Check which one has better vocabulary coverage
        coverage_with_stem = vector_with_stem.nnz
        coverage_no_stem = vector_no_stem.nnz
        
        # Use the version with better coverage
        if coverage_no_stem >= coverage_with_stem:
            email_vector = vector_no_stem
            processed_text = text_no_stem
            used_parser = "no_stemming"
        else:
            email_vector = vector_with_stem
            processed_text = text_with_stem
            used_parser = "with_stemming"
        
        # Make prediction - get probabilities
        probabilities = model.predict_proba(email_vector)[0]
        classes = model.classes_
        
        spam_prob = 0.5
        ham_prob = 0.5
        
        if len(classes) == 2:
            # Convert classes to strings for comparison
            class_list = [str(c).strip().lower() for c in classes]
            
            # Try to find spam/ham in the class names
            spam_idx = None
            ham_idx = None
            
            for i, cls in enumerate(class_list):
                if 'spam' in cls or cls == '1':
                    spam_idx = i
                elif 'ham' in cls or cls == '0':
                    ham_idx = i
            
            # If we couldn't identify, assume standard 0=ham, 1=spam
            if spam_idx is None:
                spam_idx = 1
                ham_idx = 0
            
            spam_prob = float(probabilities[spam_idx])
            ham_prob = float(probabilities[ham_idx])
        
        # Determine prediction based on probability
        is_spam = spam_prob > 0.5
        prediction_label = 'spam' if is_spam else 'ham'
        
        # Get feature importance (words with highest coefficients)
        feature_names = vectorizer.get_feature_names_out()
        email_features = email_vector.toarray()[0]
        
        # Get indices of non-zero features
        non_zero_indices = email_features.nonzero()[0]
        
        # Get model coefficients
        important_words = []
        if hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            
            # Calculate weighted importance for each word in the email
            word_importance = []
            for idx in non_zero_indices:
                word = feature_names[idx]
                coef = float(coefficients[idx])
                tf_idf_value = float(email_features[idx])
                weight = coef * tf_idf_value
                word_importance.append({
                    'word': word,
                    'weight': weight,
                    'coefficient': coef,
                    'tfidf': tf_idf_value
                })
            
            # Sort by absolute weight and take top 20
            word_importance.sort(key=lambda x: abs(x['weight']), reverse=True)
            important_words = word_importance[:20]
        
        return JsonResponse({
            'prediction': prediction_label,
            'spam_probability': spam_prob,
            'ham_probability': ham_prob,
            'important_words': important_words,
            'parsed_tokens_count': {
                'subject': len(parsed_no_stem['subject']),
                'body': len(parsed_no_stem['body'])
            },
            'debug': {
                'model_classes': [str(c) for c in classes],
                'raw_probabilities': [float(p) for p in probabilities],
                'vocabulary_coverage': {
                    'with_stemming': coverage_with_stem,
                    'no_stemming': coverage_no_stem,
                    'used_parser': used_parser
                },
                'processed_text_length': len(processed_text.split()),
                'processed_text_preview': ' '.join(processed_text.split()[:30]),
                'original_text_preview': email_content[:200],
                'total_vocabulary_size': len(feature_names),
                'non_zero_features': len(non_zero_indices)
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
