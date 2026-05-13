from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

import os
from pathlib import Path
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    HAVE_TORCH = True
except Exception as e:
    print(f"Optional dependency missing (torch/transformers): {e}")
    HAVE_TORCH = False
    torch = None
    nn = None
    F = None
    AutoTokenizer = None
    AutoModel = None

base_dir = Path(__file__).resolve().parents[2]
legacy_model = Path(__file__).resolve().parent / 'model.pkl'
legacy_vec = Path(__file__).resolve().parent / 'vectorizer.pkl'
for p in (legacy_model, legacy_vec):
    if p.exists():
        try:
            p.unlink()
            print(f"Removed legacy file: {p.name}")
        except Exception as e:
            print(f"Could not remove {p.name}: {e}")

def first_existing_path(candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


BEST_MODEL_PATH = first_existing_path([
    base_dir / 'Datacleaning' / 'outputs' / 'distillbert(FullDataset)' / 'best_model.pt',
    base_dir / 'outputs' / 'best_model.pt',
])

RF_MODEL_PATH = first_existing_path([
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'rf_model.pkl',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'rf_model.pkl',
])

RF_VECTORIZER_PATH = first_existing_path([
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'tfidf_vectorizer.joblib',
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'vectorizer.pkl',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'tfidf_vectorizer.joblib',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'vectorizer.pkl',
])
MODEL_NAME = 'distilroberta-base'

# Validate that we're using a distil model
if 'distil' not in MODEL_NAME.lower():
    raise ValueError(f"ERROR: Expected a distil-based model (distilbert/distilroberta), but got '{MODEL_NAME}'")
print(f"✓ Model validation passed: Using {MODEL_NAME}")

model = None
tokenizer = None
device = None
rf_model = None
rf_vectorizer = None


def fallback_phishing_score(text):
    keywords = ['urgent', 'password', 'bank', 'log in', 'wire', 'account', 'verify', 'click here', 'suspended', 'arrange', 'quick']
    lowered_text = text.lower()
    keyword_count = sum(1 for keyword in keywords if keyword in lowered_text)
    phishing_score = 0.15 + (keyword_count * 0.2)
    return min(0.99, phishing_score)


if HAVE_TORCH:
    class MultiTaskPhishModel(nn.Module):
        def __init__(self, model_name, dropout, manip_dim):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(model_name)
            hidden_size = self.backbone.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, 2)
            self.manip_head = nn.Linear(hidden_size, manip_dim)
            self.anom_head = nn.Linear(hidden_size, 1)

        def forward(self, input_ids, attention_mask):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
            manip = torch.sigmoid(self.manip_head(pooled))
            anom = torch.sigmoid(self.anom_head(pooled))
            return logits, manip, anom

    # Try to load the trained PyTorch checkpoint and tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if BEST_MODEL_PATH.exists():
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = MultiTaskPhishModel(MODEL_NAME, dropout=0.3, manip_dim=5).to(device)
            model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
            model.eval()
            print(f"SUCCESS: Loaded PyTorch checkpoint from {BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            model = None
            tokenizer = None
    else:
        print(f"WARNING: {BEST_MODEL_PATH} not found. Falling back to dummy predictor.")
        model = None
        tokenizer = None
else:
    print("Torch/Transformers not available — server will run using dummy fallback.")

if RF_MODEL_PATH.exists():
    try:
        with open(RF_MODEL_PATH, 'rb') as f:
            rf_model = pickle.load(f)
        print(f"SUCCESS: Loaded Random Forest model from {RF_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load Random Forest model: {e}")
        rf_model = None
else:
    print(f"WARNING: {RF_MODEL_PATH} not found. Random Forest inference unavailable.")

if RF_VECTORIZER_PATH.exists():
    try:
        if RF_VECTORIZER_PATH.suffix.lower() == '.joblib':
            import joblib
            rf_vectorizer = joblib.load(RF_VECTORIZER_PATH)
        else:
            with open(RF_VECTORIZER_PATH, 'rb') as f:
                rf_vectorizer = pickle.load(f)
        print(f"SUCCESS: Loaded Random Forest vectorizer from {RF_VECTORIZER_PATH}")
    except Exception as e:
        print(f"Failed to load Random Forest vectorizer: {e}")
        rf_vectorizer = None
else:
    print(f"WARNING: {RF_VECTORIZER_PATH} not found. Random Forest inference unavailable.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text_to_test = data.get('text', '')
    model_choice = str(data.get('model', 'distilbert')).strip().lower()

    if not text_to_test:
        return jsonify({'error': 'No text provided'}), 400

    if model_choice in ('rf', 'random_forest', 'randomforest'):
        selected_model = 'random_forest'
        if rf_model is not None and rf_vectorizer is not None:
            try:
                features = rf_vectorizer.transform([str(text_to_test)])
                if hasattr(rf_model, 'predict_proba'):
                    phishing_score = float(rf_model.predict_proba(features)[0][1])
                else:
                    pred = int(rf_model.predict(features)[0])
                    phishing_score = float(pred)
                result = "Phishing" if phishing_score >= 0.60 else "Safe"
                confidence = phishing_score
                model_fallback = False
            except Exception as e:
                print(f"Random Forest inference failed: {e}")
                model_fallback = True
        else:
            model_fallback = True
    else:
        selected_model = 'distilbert'
        if model is not None and tokenizer is not None:
            try:
                enc = tokenizer([str(text_to_test)], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                with torch.no_grad():
                    logits, mp, anom = model(input_ids, attention_mask)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                phishing_score = float(probs[0]) if hasattr(probs, '__len__') else float(probs)
                result = "Phishing" if phishing_score >= 0.60 else "Safe"
                confidence = phishing_score
                model_fallback = False
            except Exception as e:
                print(f"Distil model inference failed: {e}")
                model_fallback = True
        else:
            model_fallback = True

    if model_fallback:
        # Deterministic fallback so the same text always gets the same result.
        phishing_score = fallback_phishing_score(text_to_test)
        result = "Phishing" if phishing_score >= 0.60 else "Safe"
        confidence = float(phishing_score)

    return jsonify({
        'prediction': result,
        'confidence': float(confidence),
        'original_text_length': len(text_to_test),
        'model_used': selected_model,
        'fallback_used': model_fallback,
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
