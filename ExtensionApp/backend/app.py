from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

import os
from pathlib import Path
import re
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

def extract_email_from_header(header_text):
    """Return the first email address found in the header text, or empty string."""
    if not isinstance(header_text, str):
        return ""
    m = re.search(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', header_text)
    return m.group(0).strip() if m else ""


def sender_name_from_email(email_text):
    if not isinstance(email_text, str) or '@' not in email_text:
        return ""
    local = email_text.split('@', 1)[0]
    parts = [part for part in re.split(r'[._\-]+', local) if part]
    if not parts:
        return ""
    return " ".join(part.capitalize() for part in parts)


def extract_signature_name(body_text):
    """Try to extract a signer name from common sign-off patterns in the body.

    Looks for lines after sign-offs like 'Best regards', 'Regards', 'Sincerely',
    'Thanks', 'Thank you', and returns the following name line if present.
    """
    if not isinstance(body_text, str):
        return ""

    # Normalize line endings
    text = body_text.replace('\r', '')

    # Common signoff keywords
    signoffs = r'(?:best regards|regards|sincerely|kind regards|thanks|thank you|cheers|best)'

    # Pattern: signoff followed by optional comma and then a newline and a capitalized name line
    signoff_newline = re.compile(rf'(?im){signoffs}[\s,:\-]*\n+\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){{0,3}})')
    m = signoff_newline.search(text)
    if m:
        return m.group(1).strip()

    # Also handle inline signoffs like 'Best regards, Bill Gates' where the name is multi-word
    pattern_inline = re.compile(rf'(?im){signoffs}\s*,?\s*([A-Z][A-Za-z\'\-]+(?:\s+[A-Z][A-Za-z\'\-]+)+)')
    m2 = pattern_inline.search(text)
    if m2:
        return m2.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tail = lines[-6:] if len(lines) >= 6 else lines
    name_pattern = re.compile(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}$')
    # skip common signoff words and look for a capitalized name line
    skip = set([s.lower() for s in ['best regards', 'regards', 'sincerely', 'kind regards', 'thanks', 'thank you', 'cheers', 'best']])
    for ln in reversed(tail):
        if ln.lower() in skip:
            continue
        if name_pattern.match(ln):
            return ln.strip()

    return ""


def extract_header_body(text):
    """Split combined text into header and body parts."""
    if not isinstance(text, str):
        return "", ""
    parts = re.split(r'\n\s*\n', text, maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text[:200], text[200:]

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

    # continue to assemble full response including sender checks

    # --- Sender/signature consistency checks ---
    try:
        # Search whole text for header email and signature (more robust)
        full_text = str(text_to_test)
        from_match = re.search(r'From:\s*([^\n<]+)', full_text, re.IGNORECASE)
        header_sender_raw = from_match.group(1).strip() if from_match else ''
        header_email = extract_email_from_header(full_text)
        if header_sender_raw.strip().lower() == 'me' and header_email:
            header_sender_raw = sender_name_from_email(header_email) or header_email
        signature_name = extract_signature_name(full_text)
        if not signature_name:
            # Fallback: last multi-word capitalized name in the text
            names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', full_text)
            if names:
                signature_name = names[-1]

        def _normalize(n):
            return re.sub(r'[^a-z]', '', (n or '').lower())

        header_name_norm = _normalize(header_sender_raw)
        signature_norm = _normalize(signature_name)
        email_local = (header_email.split('@')[0] if header_email else '')
        email_local_norm = _normalize(email_local)

        name_matches_email = False
        name_matches_signature = False
        signature_matches_email = False
        if header_name_norm and email_local_norm:
            name_matches_email = (header_name_norm in email_local_norm) or (email_local_norm in header_name_norm)
        if signature_norm and header_name_norm:
            name_matches_signature = (signature_norm in header_name_norm) or (header_name_norm in signature_norm)
        if signature_norm and email_local_norm:
            signature_matches_email = (signature_norm in email_local_norm) or (email_local_norm in signature_norm)

        sender_info = {
            'header_sender': header_sender_raw,
            'header_email': header_email,
            'signature_name': signature_name,
            'name_matches_email': name_matches_email,
            'name_matches_signature': name_matches_signature,
            'suspicious_sender_mismatch': False
        }

        # Flag as suspicious when signature doesn't match header email/local-part
        if (signature_name and header_email and not signature_matches_email) or (header_email and header_sender_raw and not name_matches_email):
            sender_info['suspicious_sender_mismatch'] = True
        sender_info['signature_matches_email'] = signature_matches_email

        # If sender-signature mismatch found, enforce Phishing decision
        enforced = False
        if sender_info.get('suspicious_sender_mismatch'):
            enforced = True
            result = 'Phishing'
            confidence = max(float(confidence), 0.95)

        # Attach sender_info to response
        resp = {
            'prediction': result,
            'confidence': float(confidence),
            'original_text_length': len(text_to_test),
            'model_used': selected_model,
            'fallback_used': model_fallback,
            'sender_info': sender_info,
            'enforced_sender_mismatch': enforced
        }
        return jsonify(resp)
    except Exception:
        # If anything goes wrong, return the original response
        return jsonify({
            'prediction': result,
            'confidence': float(confidence),
            'original_text_length': len(text_to_test),
            'model_used': selected_model,
            'fallback_used': model_fallback,
        })
if __name__ == '__main__':
    app.run(debug=True, port=5000)