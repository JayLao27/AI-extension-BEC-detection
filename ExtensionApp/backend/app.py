from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allow cross-origin requests from the Chrome Extension

# Load your trained machine learning model here
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
vec_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

if os.path.exists(model_path) and os.path.exists(vec_path):
    model = pickle.load(open(model_path, 'rb'))
    vectorizer = pickle.load(open(vec_path, 'rb'))
    print("SUCCESS: Loaded real machine learning models!")
else:
    model = None
    vectorizer = None
    print("WARNING: model.pkl and/or vectorizer.pkl not found. Please train and save your models first.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text_to_test = data.get('text', '')

    if not text_to_test:
        return jsonify({'error': 'No text provided'}), 400

    if model and vectorizer:
        # Use your trained model for prediction
        # Ensure we only pass valid strings to vectorizer
        cleaned_text = str(text_to_test) if text_to_test is not None else ""
        features = vectorizer.transform([cleaned_text])
        
        # Calculate phishing score between 0.0 and 1.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            # Verify shape before indexing to prevent out of bounds
            phishing_score = probs[1] if len(probs) > 1 else probs[0]
        else:
            prediction = model.predict(features)
            # Safely extract prediction regardless of shape
            pred_val = prediction[0] if (hasattr(prediction, '__len__') and not isinstance(prediction, str)) else prediction
            phishing_score = 0.95 if int(pred_val) == 1 else 0.15
            
        result = "Phishing" if phishing_score >= 0.60 else "Safe"
        confidence = float(phishing_score) # Display the phishing probability directly
    else:
        # Dummy logic fallback until models are exported
        import random
        # Check for suspicious keywords
        keywords = ['urgent', 'password', 'bank', 'log in', 'wire', 'account', 'verify', 'click here', 'suspended', 'arrange', 'quick']
        keyword_count = sum(1 for keyword in keywords if keyword in text_to_test.lower())
        
        # Calculate a simulated phishing score from 0.0 to 1.0 (0 to 100%)
        # Give a base score between 10% and 40%, plus 20% for each keyword found
        phishing_score = random.uniform(0.1, 0.4) + (keyword_count * 0.2)
        phishing_score = min(0.99, phishing_score) # Cap at 99%
        
        # >= 60% is categorized as Phishing
        result = "Phishing" if phishing_score >= 0.60 else "Safe"
        confidence = float(phishing_score) # ensure float for JSON serialization

    return jsonify({
        'prediction': result,
        'confidence': float(confidence),
        'original_text_length': len(text_to_test)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
