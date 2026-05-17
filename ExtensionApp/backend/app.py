from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import json
import joblib
from pathlib import Path

app = Flask(__name__)
CORS(app) 

base_dir = Path(__file__).resolve().parents[2]

# ============================================================================
# TWO-STAGE BEC DETECTION MODELS
# ============================================================================
# Models: Header (XGBoost) + Body (KNN) + Final (Logistic Regression)

# Define paths for Two-Stage models
MODELS_DIR = Path(__file__).resolve().parent / 'models'
HEADER_VECTORIZER_PATH = MODELS_DIR / 'header_vectorizer.joblib'
BODY_VECTORIZER_PATH = MODELS_DIR / 'body_vectorizer.joblib'
HEADER_MODEL_PATH = MODELS_DIR / 'header_model.joblib'
BODY_MODEL_PATH = MODELS_DIR / 'body_model.joblib'
FINAL_MODEL_PATH = MODELS_DIR / 'final_model.joblib'
METADATA_PATH = MODELS_DIR / 'metadata.json'

# Initialize model containers
header_vectorizer = None
body_vectorizer = None
header_model = None
body_model = None
final_model = None
metadata = {}

print("\n" + "="*60)
print("INITIALIZING TWO-STAGE BEC DETECTION SYSTEM")
print("="*60)

# Load metadata
if METADATA_PATH.exists():
    try:
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Loaded metadata: {metadata.get('description', 'Two-Stage BEC Detection')}")
    except Exception as e:
        print(f"⚠ Failed to load metadata: {e}")

# BEC keyword list used for heuristic signals and fallback scoring
# Includes multiple languages and common phrases
BEC_KEYWORDS = [
    'invoice','payment','paycheck','transfer','bank statement','bank details',
    'closing','funds','bank account','account details','remittance','purchase',
    'deposit','PO#','Zahlung','Rechnung','Paiement','virement bancaire',
    'Bankuberweisung','hacked','phishing'
]

# Load vectorizers
if HEADER_VECTORIZER_PATH.exists():
    try:
        header_vectorizer = joblib.load(HEADER_VECTORIZER_PATH)
        print(f"✓ Loaded header_vectorizer ({header_vectorizer.get_feature_names_out().shape[0]} features)")
    except Exception as e:
        print(f"✗ Failed to load header_vectorizer: {e}")

if BODY_VECTORIZER_PATH.exists():
    try:
        body_vectorizer = joblib.load(BODY_VECTORIZER_PATH)
        print(f"✓ Loaded body_vectorizer ({body_vectorizer.get_feature_names_out().shape[0]} features)")
    except Exception as e:
        print(f"✗ Failed to load body_vectorizer: {e}")

# Load models
if HEADER_MODEL_PATH.exists():
    try:
        header_model = joblib.load(HEADER_MODEL_PATH)
        print(f"✓ Loaded header_model ({metadata.get('header_model', 'unknown')})")
    except Exception as e:
        print(f"✗ Failed to load header_model: {e}")

if BODY_MODEL_PATH.exists():
    try:
        body_model = joblib.load(BODY_MODEL_PATH)
        print(f"✓ Loaded body_model ({metadata.get('body_model', 'unknown')})")
    except Exception as e:
        print(f"✗ Failed to load body_model: {e}")

if FINAL_MODEL_PATH.exists():
    try:
        final_model = joblib.load(FINAL_MODEL_PATH)
        print(f"✓ Loaded final_model (Logistic Regression Stacking)")
    except Exception as e:
        print(f"✗ Failed to load final_model: {e}")

models_ready = all([
    header_vectorizer is not None,
    body_vectorizer is not None,
    header_model is not None,
    body_model is not None,
    final_model is not None
])

if models_ready:
    print(f"\n✓ TWO-STAGE SYSTEM READY")
    print(f"  Header Accuracy: {metadata.get('header_accuracy', 'N/A')}")
    print(f"  Body Accuracy: {metadata.get('body_accuracy', 'N/A')}")
    print(f"  Final Accuracy: {metadata.get('final_accuracy', 'N/A')}")
else:
    print(f"\n⚠ WARNING: Some models failed to load. System will use fallback.")
print("="*60 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_header_body(text):
    """
    Separate email header from body.
    Header contains: From, Subject, Reply-To, Date, etc.
    Body contains: the sender's actual message
    """
    if not isinstance(text, str):
        return "", ""
    
    # Standard raw emails use \n\n to separate headers and body
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    
    # Fallback heuristic: assume first 150 chars contains header info
    return text[:150], text[150:]

def extract_sender_from_header(header_text):
    """
    Extract sender (From) email/name from header.
    """
    if not isinstance(header_text, str):
        return ""

    # Look for From: field
    from_match = re.search(r'From:\s*(.+?)(?:\n|$)', header_text, re.IGNORECASE)
    if from_match:
        raw = from_match.group(1).strip()
        try:
            from email.utils import parseaddr
            name, email_addr = parseaddr(raw)
            if name:
                return name.strip().strip('"')
            if email_addr:
                return email_addr.split('@')[0]
        except Exception:
            # Fallback: remove angle-bracketed addresses and return text
            return re.sub(r'<[^>]+>', '', raw).strip()

    # Fallback: first line might be sender; clean angle-bracketed addresses
    first_line = header_text.split('\n')[0] if header_text else ""
    return re.sub(r'<[^>]+>', '', first_line).strip()


def extract_subject_from_header(header_text):
    """
    Extract the Subject line from the header and return cleaned text
    (removes any <email@domain> parts).
    """
    if not isinstance(header_text, str):
        return ""

    subj_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', header_text, re.IGNORECASE)
    if subj_match:
        raw = subj_match.group(1).strip()
    else:
        # Try a safe fallback scanning lines
        raw = ""
        for line in header_text.split('\n'):
            if line.lower().startswith('subject:'):
                raw = line.split(':', 1)[1].strip()
                break

    # Remove any angle-bracketed content like <receiver@gmail.com>
    cleaned = re.sub(r'<[^>]+>', '', raw).strip()
    return cleaned

def fallback_bec_score(text):
    """
    Fallback BEC detection using keyword heuristics.
    """
    lowered_text = text.lower()
    keyword_count = sum(1 for kw in BEC_KEYWORDS if kw.lower() in lowered_text)
    bec_score = 0.05 + (keyword_count * 0.12)
    return min(0.99, bec_score)

def predict_bec_two_stage(text):
    """
    Two-stage BEC detection CASCADE logic:
    
    Stage 1: Header Model (Suspicious Sender Detection)
    - Analyzes sender, reply-to mismatches, etc.
    - If header looks LEGITIMATE -> Return LEGITIMATE (skip body)
    - If header looks SUSPICIOUS -> Proceed to Stage 2
    
    Stage 2: Body Model (Message Content Analysis)
    - Analyzes urgency, financial keywords, threats, etc.
    - If body also looks SUSPICIOUS -> Return BEC (phishing)
    - If body looks LEGITIMATE -> Return LEGITIMATE
    
    Returns: (prediction_label, confidence_score, details_dict)
    """
    if not models_ready:
        return "Unknown", fallback_bec_score(text), {}
    
    try:
        import numpy as np

        # Extract header and body
        header, body = extract_header_body(text)

        # Extract sender information from header
        sender = extract_sender_from_header(header)
        # Extract cleaned subject (removes <...> email addresses)
        subject = extract_subject_from_header(header)

        # Ensure non-empty
        if not header.strip():
            header = text[:200]
        if not body.strip():
            body = text

        # ===== STAGE 1: HEADER ANALYSIS (Sender Verification) =====
        header_features = header_vectorizer.transform([header])
        header_pred_proba = header_model.predict_proba(header_features)[0]
        header_bec_score = float(header_pred_proba[1])  # Probability of suspicious sender
        header_prediction_label = "Suspicious" if header_bec_score >= 0.5 else "Legitimate"

        # ===== STAGE 2: BODY ANALYSIS (Message Content) =====
        body_features = body_vectorizer.transform([body])
        body_pred_proba = body_model.predict_proba(body_features)[0]
        body_bec_score = float(body_pred_proba[1])  # Probability of suspicious content
        body_prediction_label = "Suspicious" if body_bec_score >= 0.5 else "Legitimate"

        # Keyword matches in header and body
        header_lower = header.lower()
        body_lower = body.lower()
        subject_lower = subject.lower() if subject else ""

        # Match keywords against header+subject and body separately
        header_keyword_matches = [kw for kw in BEC_KEYWORDS if kw.lower() in (header_lower + ' ' + subject_lower)]
        body_keyword_matches = [kw for kw in BEC_KEYWORDS if kw.lower() in body_lower]
        subject_keyword_matches = [kw for kw in BEC_KEYWORDS if kw.lower() in subject_lower]
        header_keyword_count = len(header_keyword_matches)
        body_keyword_count = len(body_keyword_matches)
        total_keyword_matches = header_keyword_count + body_keyword_count

        # Small heuristic boost from keyword matches (capped)
        keyword_signal = min(0.2, 0.05 * total_keyword_matches)

        header_is_suspicious = header_bec_score >= 0.5
        body_is_suspicious = body_bec_score >= 0.5

        # ===== FINAL DECISION: CASCADE LOGIC =====
        # Both header AND body must be suspicious to classify as BEC normally,
        # but keyword evidence can increase confidence.
        if header_is_suspicious and body_is_suspicious:
            # Stack both scores for final confidence
            stacked = np.array([[header_bec_score, body_bec_score]])
            final_pred_proba = final_model.predict_proba(stacked)[0]
            final_bec_score = float(final_pred_proba[1])
            final_bec_score = min(0.99, final_bec_score + keyword_signal)
            prediction = "BEC"
        else:
            # If either header or body is clean, it's legitimate but include keyword signal
            final_bec_score = (min(header_bec_score, body_bec_score) * 0.5) + keyword_signal
            final_bec_score = min(0.99, final_bec_score)
            prediction = "Legitimate"

        details = {
            "sender": sender,
            "subject": subject,
            "subject_keyword_matches": subject_keyword_matches,
            "subject_keyword_count": len(subject_keyword_matches),
            "header_prediction": header_prediction_label,
            "header_bec_score": round(header_bec_score, 4),
            "header_keyword_matches": header_keyword_matches,
            "header_keyword_count": header_keyword_count,
            "stage_1_header_suspicious": header_is_suspicious,
            "body_prediction": body_prediction_label,
            "body_bec_score": round(body_bec_score, 4),
            "body_keyword_matches": body_keyword_matches,
            "body_keyword_count": body_keyword_count,
            "stage_2_body_suspicious": body_is_suspicious,
            "final_bec_score": round(final_bec_score, 4),
            "keyword_signal": round(keyword_signal, 4),
            "header_model": metadata.get('header_model', 'XGBoost'),
            "body_model": metadata.get('body_model', 'KNN'),
            "cascade_logic": "Both header AND body must be suspicious; keywords add heuristic signal"
        }

        return prediction, final_bec_score, details
    
    except Exception as e:
        print(f"Error in two-stage prediction: {e}")
        return "Error", fallback_bec_score(text), {"error": str(e)}

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for BEC detection.
    
    Expected input:
    {
        "text": "email content here"
    }
    
    Returns:
    {
        "prediction": "BEC" or "Legitimate",
        "confidence": 0.0-1.0,
        "model_used": "two_stage",
        "details": {...}
    }
    """
    data = request.json
    text_to_test = data.get('text', '')

    if not text_to_test:
        return jsonify({'error': 'No text provided'}), 400

    # Run two-stage BEC detection
    prediction, confidence, details = predict_bec_two_stage(text_to_test)
    
    return jsonify({
        'prediction': prediction,
        'confidence': float(confidence),
        'bec_probability': f"{float(confidence)*100:.2f}%",
        'original_text_length': len(text_to_test),
        'model_used': 'two_stage_bec',
        'system': 'Two-Stage BEC Detection (Header + Body)',
        'fallback_used': not models_ready,
        'details': details,
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok' if models_ready else 'degraded',
        'models_ready': models_ready,
        'header_model': metadata.get('header_model'),
        'body_model': metadata.get('body_model'),
        'final_accuracy': metadata.get('final_accuracy'),
    })

@app.route('/verify-models', methods=['GET'])
def verify_models():
    """
    Verify all models are loaded and working correctly.
    Returns detailed model information and test prediction.
    """
    if not models_ready:
        return jsonify({
            'status': 'error',
            'message': 'Models not ready',
            'models_loaded': {
                'header_vectorizer': header_vectorizer is not None,
                'body_vectorizer': body_vectorizer is not None,
                'header_model': header_model is not None,
                'body_model': body_model is not None,
                'final_model': final_model is not None,
            }
        }), 503
    
    try:
        # Test with a sample email
        test_email = """From: boss@company.com
Reply-To: someone@external-bank.com
Subject: Urgent: Wire Transfer Needed

Hi,
Please wire $50,000 immediately to the account below for a confidential business transaction.
This is urgent and needs to be done before end of day.
Don't discuss with anyone.

Best regards,
CEO"""
        
        # Run prediction
        prediction, confidence, details = predict_bec_two_stage(test_email)
        
        return jsonify({
            'status': 'ok',
            'message': 'All models loaded and verified',
            'models_configuration': {
                'header_model': metadata.get('header_model'),
                'body_model': metadata.get('body_model'),
                'header_features': header_vectorizer.get_feature_names_out().shape[0],
                'body_features': body_vectorizer.get_feature_names_out().shape[0],
            },
            'model_accuracy': {
                'header_accuracy': metadata.get('header_accuracy'),
                'body_accuracy': metadata.get('body_accuracy'),
                'final_accuracy': metadata.get('final_accuracy'),
            },
            'test_prediction': {
                'prediction': prediction,
                'confidence': float(confidence),
                'details': details
            },
            'metadata': metadata
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Verification failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
