from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

import os
import json
from pathlib import Path
import re
import pickle
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    from xgboost import XGBClassifier
    HAVE_TORCH = True
except Exception as e:
    print(f"Optional dependency missing (torch/transformers): {e}")
    HAVE_TORCH = False
    torch = None
    nn = None
    F = None
    AutoTokenizer = None
    AutoModel = None
    XGBClassifier = None

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

BEST_MODEL_INFO_PATH = first_existing_path([
    base_dir / 'outputs' / 'distilbert_header_body' / 'best_model_info.json',
    base_dir / 'Datacleaning' / 'outputs' / 'distilbert_header_body' / 'best_model_info.json',
])

RF_MODEL_PATH = first_existing_path([
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'rf_model.pkl',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'rf_model.pkl',
    base_dir / 'outputs' / 'distilbert_header_body' / 'rf_model.joblib',
])

RF_VECTORIZER_PATH = first_existing_path([
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'tfidf_vectorizer.joblib',
    base_dir / 'Datacleaning' / 'outputs' / 'RandomForest(FullDataset)' / 'vectorizer.pkl',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'tfidf_vectorizer.joblib',
    base_dir / 'outputs' / 'RandomForest(FullDataset)' / 'vectorizer.pkl',
    base_dir / 'outputs' / 'distilbert_header_body' / 'tfidf_vectorizer.joblib',
])
XGB_MODEL_PATH = first_existing_path([
    base_dir / 'outputs' / 'distilbert_header_body' / 'xgb_model.joblib',
    base_dir / 'Datacleaning' / 'outputs' / 'distilbert_header_body' / 'xgb_model.joblib',
])

MODEL_NAME = 'distilbert-base-uncased'

best_model_info = {}
if BEST_MODEL_INFO_PATH.exists():
    try:
        with open(BEST_MODEL_INFO_PATH, 'r', encoding='utf-8') as f:
            best_model_info = json.load(f)
        print(f"Loaded best-model metadata from {BEST_MODEL_INFO_PATH}")
    except Exception as e:
        print(f"Failed to load best-model metadata: {e}")
        best_model_info = {}

# Validate that we're using a distil model
if 'distil' not in MODEL_NAME.lower():
    raise ValueError(f"ERROR: Expected a distil-based model (distilbert/distilroberta), but got '{MODEL_NAME}'")
print(f"[OK] Model validation passed: Using {MODEL_NAME}")

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
            self.header_backbone = AutoModel.from_pretrained(model_name)
            self.body_backbone = AutoModel.from_pretrained(model_name)
            hidden_size = self.header_backbone.config.hidden_size + self.body_backbone.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, 2)
            self.manip_head = nn.Linear(hidden_size, manip_dim)
            self.anom_head = nn.Linear(hidden_size, 1)

        def forward(self, h_input_ids, h_attention_mask, b_input_ids, b_attention_mask):
            h_out = self.header_backbone(input_ids=h_input_ids, attention_mask=h_attention_mask)
            b_out = self.body_backbone(input_ids=b_input_ids, attention_mask=b_attention_mask)
            pooled = torch.cat([h_out.last_hidden_state[:, 0], b_out.last_hidden_state[:, 0]], dim=1)
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
            manip = torch.sigmoid(self.manip_head(pooled))
            anom = torch.sigmoid(self.anom_head(pooled))
            return logits, manip, anom

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if best_model_info:
        print(f"Best-model metadata: {best_model_info}")

    distilbert_model = None
    distilbert_tokenizer = None
    distilbert_checkpoint = Path(best_model_info.get('best_artifact', BEST_MODEL_PATH)) if best_model_info.get('best_model', '').lower() == 'distilbert' else BEST_MODEL_PATH
    if distilbert_checkpoint.exists():
        try:
            distilbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            distilbert_model = MultiTaskPhishModel(MODEL_NAME, dropout=0.1, manip_dim=5).to(device)
            distilbert_model.load_state_dict(torch.load(distilbert_checkpoint, map_location=device))
            distilbert_model.eval()
            print(f"SUCCESS: Loaded DistilBERT checkpoint from {distilbert_checkpoint}")
        except Exception as e:
            print(f"Failed to load PyTorch model: {e}")
            distilbert_model = None
            distilbert_tokenizer = None
    else:
        print(f"WARNING: {distilbert_checkpoint} not found. Falling back to dummy predictor.")

    model = distilbert_model
    tokenizer = distilbert_tokenizer
else:
    print("Torch/Transformers not available — server will run using dummy fallback.")

rf_model = None
if RF_MODEL_PATH.exists():
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"SUCCESS: Loaded Random Forest model from {RF_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load Random Forest model: {e}")
        rf_model = None
else:
    print(f"WARNING: {RF_MODEL_PATH} not found. Random Forest inference unavailable.")

if XGB_MODEL_PATH.exists():
    try:
        xgb_model = joblib.load(XGB_MODEL_PATH)
        print(f"SUCCESS: Loaded XGBoost model from {XGB_MODEL_PATH}")
    except Exception as e:
        print(f"Failed to load XGBoost model: {e}")
        xgb_model = None
else:
    print(f"WARNING: {XGB_MODEL_PATH} not found. XGBoost inference unavailable.")

if RF_VECTORIZER_PATH.exists():
    try:
        rf_vectorizer = joblib.load(RF_VECTORIZER_PATH)
        print(f"SUCCESS: Loaded Random Forest vectorizer from {RF_VECTORIZER_PATH}")
    except Exception as e:
        print(f"Failed to load Random Forest vectorizer: {e}")
        rf_vectorizer = None
else:
    print(f"WARNING: {RF_VECTORIZER_PATH} not found. Random Forest inference unavailable.")

# ---------------------------------------------------------
# USENIX Security '19 (sec19) TWO-STAGE MODEL CASCADE
# ---------------------------------------------------------
import numpy as np

sec19_header_model = None
sec19_body_model = None
sec19_final_model = None
sec19_body_vec = None
sec19_extractor = None

models_dir = Path(__file__).resolve().parent / 'models'
SEC19_HEADER_MODEL_PATH = models_dir / 'header_model.joblib'
SEC19_BODY_MODEL_PATH = models_dir / 'body_model.joblib'
SEC19_FINAL_MODEL_PATH = models_dir / 'final_model.joblib'
SEC19_BODY_VEC_PATH = models_dir / 'body_vectorizer.joblib'
SEC19_EXTRACTOR_PATH = models_dir / 'header_extractor.joblib'

datacleaning_dir = base_dir / 'Datacleaning'
if SEC19_EXTRACTOR_PATH.exists():
    try:
        import sys
        if str(datacleaning_dir) not in sys.path:
            sys.path.insert(0, str(datacleaning_dir))
        from header_feature_extractor import HeaderFeatureExtractor
        
        sec19_extractor = joblib.load(SEC19_EXTRACTOR_PATH)
        sec19_header_model = joblib.load(SEC19_HEADER_MODEL_PATH)
        sec19_body_model = joblib.load(SEC19_BODY_MODEL_PATH)
        sec19_final_model = joblib.load(SEC19_FINAL_MODEL_PATH)
        sec19_body_vec = joblib.load(SEC19_BODY_VEC_PATH)
        print("SUCCESS: Loaded USENIX Security '19 (sec19) Two-Stage model cascade.")
    except Exception as e:
        print(f"Failed to load sec19 Two-Stage model cascade: {e}")
else:
    print(f"WARNING: sec19 cascade models not found at {SEC19_EXTRACTOR_PATH}")

def preprocess_body_text(text):
    """
    Preprocess body text based on USENIX Security '19 (sec19) baseline:
    - Remove salutations ('Dear', 'Hi', 'Hello')
    - Remove footers and common signatures ('Best regards', 'Sincerely', 'Thanks')
    - Convert to lowercase
    """
    if not isinstance(text, str):
        return ""
    text = text.replace('\r', '').lower()
    text = re.sub(r'^(?:hi|dear|hello|greetings|hey)(?:\s+\w+){0,3}[\s,]*\n+', '', text)
    signoffs = r'(?:best regards|regards|sincerely|kind regards|thanks|thank you|cheers|best|respectfully)'
    text = re.sub(rf'{signoffs}[\s,:\-]*\n+.*$', '', text, flags=re.DOTALL)
    return text.strip()


def build_explanation(prediction, model_used, features=None, s1_prob=None, s2_prob=None,
                      final_prob=None, impersonation_score=None, ml_score=None,
                      sender_email=None, sender_name=None, reply_to=None,
                      domain=None, name_matches_email=None, reply_to_mismatch=None,
                      sender_rarity=None, signature_mismatch=None, fallback_used=False):
    """
    Build a human-readable explanation and list of evidence for why an email was
    classified as Safe or Phishing, based on the feature values computed at runtime.

    Returns a dict with:
      - summary: one-sentence verdict
      - reasons: list of plain-English bullet points explaining each signal
      - stage_breakdown: (optional) dict with per-stage probabilities for transparency
    """
    reasons = []
    stage_breakdown = {}

    # ---------------------------------------------------------------
    # STAGE 1 / HEADER-BASED SIGNALS
    # ---------------------------------------------------------------
    common_providers = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}

    # Reply-To mismatch
    if reply_to_mismatch is not None:
        if reply_to_mismatch == 1 or reply_to_mismatch is True:
            reasons.append(
                f"Reply-To address ({reply_to}) differs from the sender address ({sender_email}). "
                "This is a classic BEC trick to redirect replies to an attacker-controlled inbox."
            )
        else:
            reasons.append(
                f"Reply-To address matches the sender address ({sender_email}), "
                "which is consistent with legitimate correspondence."
            )

    # Free/public email domain
    if domain and domain in common_providers:
        reasons.append(
            f"Sender uses a free public email provider ({domain}). "
            "Executives and companies rarely send business-critical emails from personal webmail accounts."
        )
    elif domain and domain:
        reasons.append(
            f"Sender domain ({domain}) appears to be a corporate or custom domain, "
            "reducing the likelihood of spoofing via free webmail."
        )

    # Name vs email mismatch
    if name_matches_email is not None:
        if name_matches_email == 0 or name_matches_email is False:
            if sender_name and sender_email:
                reasons.append(
                    f"Display name \"{sender_name}\" does not correspond to the email address "
                    f"<{sender_email}>. Attackers often forge the display name to impersonate "
                    "an executive while hiding their real email."
                )
        else:
            reasons.append(
                f"Display name \"{sender_name}\" is consistent with the email address "
                f"<{sender_email}>, suggesting no name spoofing."
            )

    # Sender rarity / historical frequency
    if sender_rarity is not None:
        if sender_rarity >= 0.9:
            reasons.append(
                f"This sender/email combination has never been seen before in historical data "
                "(rarity score: {:.2f}/1.0). First-time senders are a strong BEC signal — "
                "legitimate business contacts appear repeatedly over time.".format(sender_rarity)
            )
        elif sender_rarity >= 0.5:
            reasons.append(
                "This sender has appeared only a few times in historical data "
                "(rarity score: {:.2f}/1.0), which is mildly suspicious.".format(sender_rarity)
            )
        else:
            reasons.append(
                "This sender is a well-known, frequently observed address in historical data "
                "(rarity score: {:.2f}/1.0), strongly suggesting a legitimate sender.".format(sender_rarity)
            )

    # Signature mismatch
    if signature_mismatch:
        reasons.append(
            "The name in the email signature does not match the sender's email address, "
            "indicating possible identity spoofing."
        )

    # Stage 1 probability
    if s1_prob is not None:
        stage_breakdown['stage_1_impersonation_probability'] = round(s1_prob, 4)
        if s1_prob < 0.5:
            reasons.append(
                f"Stage 1 header analysis gave a low impersonation risk score "
                f"({s1_prob:.1%}), so the email passed the safety gate without "
                "needing full body content analysis."
            )
        else:
            reasons.append(
                f"Stage 1 header analysis flagged this email with {s1_prob:.1%} "
                "impersonation risk — above the 50% safety threshold — "
                "triggering deeper body content analysis."
            )

    # ---------------------------------------------------------------
    # STAGE 2 / BODY-BASED SIGNALS
    # ---------------------------------------------------------------
    if s2_prob is not None:
        stage_breakdown['stage_2_body_phishing_probability'] = round(s2_prob, 4)
        if s2_prob >= 0.7:
            reasons.append(
                f"Stage 2 body content analysis found the email body highly similar to known "
                f"phishing/BEC emails in training data ({s2_prob:.1%} phishing probability). "
                "The language pattern matches wire-transfer fraud, urgency tactics, or credential theft."
            )
        elif s2_prob >= 0.4:
            reasons.append(
                f"Stage 2 body content analysis found moderate phishing signals in the email body "
                f"({s2_prob:.1%} phishing probability)."
            )
        else:
            reasons.append(
                f"Stage 2 body content analysis found the email body consistent with legitimate "
                f"business communication ({s2_prob:.1%} phishing probability)."
            )

    # ---------------------------------------------------------------
    # FINAL STACKED / COMBINED DECISION
    # ---------------------------------------------------------------
    if final_prob is not None:
        stage_breakdown['final_stacked_probability'] = round(final_prob, 4)

    if impersonation_score is not None and ml_score is not None:
        if impersonation_score >= 0.8 and ml_score >= 0.4:
            reasons.append(
                "Both the header impersonation score and body content score were elevated. "
                "When both signals agree, the combined risk is very high."
            )

    # Fallback case
    if fallback_used:
        reasons.append(
            "Primary model was unavailable; keyword-based scoring was used as a fallback."
        )

    # ---------------------------------------------------------------
    # BUILD SUMMARY SENTENCE
    # ---------------------------------------------------------------
    if prediction == 'Safe':
        summary = (
            "This email appears to be safe. "
            "The sender identity looks legitimate and the body content does not match known phishing patterns."
        )
    else:
        summary = (
            "This email is likely a Business Email Compromise (BEC) or phishing attempt. "
            "Multiple signals in the header and/or body indicate spoofing or fraudulent intent."
        )

    return {
        'summary': summary,
        'reasons': reasons,
        'stage_breakdown': stage_breakdown
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text_to_test = data.get('text', '')
    model_choice = str(data.get('model', 'distilbert')).strip().lower()

    if not text_to_test:
        return jsonify({'error': 'No text provided'}), 400

    if model_choice in ('sec19', 'two_stage', 'sec19_two_stage'):
        selected_model = 'sec19_two_stage'
        if (sec19_header_model is not None and 
            sec19_body_model is not None and 
            sec19_final_model is not None and 
            sec19_body_vec is not None and 
            sec19_extractor is not None):
            try:
                full_text = str(text_to_test)
                
                # 1. Parse Header (capture full From line so "Name <email>" formats work)
                from_match = re.search(r'From:\s*(.+?)(?:\n|$)', full_text, re.IGNORECASE)
                sender_raw = from_match.group(1).strip() if from_match else ''
                sender_email = extract_email_from_header(full_text)
                
                if sender_raw.strip().lower() == 'me' and sender_email:
                    sender_raw = sender_name_from_email(sender_email) or sender_email
                
                # Reply-To parsing
                reply_to_match = re.search(r'Reply-To:\s*(.+?)(?:\n|$)', full_text, re.IGNORECASE)
                reply_to = reply_to_match.group(1).strip() if reply_to_match else sender_email
                
                # Subject parsing
                subj_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', full_text, re.IGNORECASE)
                subject = subj_match.group(1).strip() if subj_match else ''
                
                # Extract features using extractor
                features = sec19_extractor.extract_header_features(
                    sender=sender_raw if sender_raw else sender_email,
                    reply_to=reply_to,
                    subject=subject,
                    is_training=False
                )
                
                # Features vector for Stage 1 RF
                s1_features = np.array([[
                    features['reply_to_mismatch'],
                    features['name_email_mismatch'],
                    features['sender_rarity']
                ]])
                
                # stage 1 prob (phishing probability)
                s1_prob = float(sec19_header_model.predict_proba(s1_features)[0][1])
                
                # Sequential Impersonation Gate
                impersonation_threshold = 0.5
                
                signature_name = extract_signature_name(full_text)
                if not signature_name:
                    # Fallback: last multi-word capitalized name in the text
                    names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', full_text)
                    if names:
                        signature_name = names[-1]
                
                # Update sender_info
                sender_info = {
                    'header_sender': features['sender_name'],
                    'header_email': features['sender_email'],
                    'signature_name': signature_name or features['sender_name'],
                    'name_matches_email': features['name_email_mismatch'] == 0,
                    'name_matches_signature': True, # mock/default
                    'suspicious_sender_mismatch': features['name_email_mismatch'] > 0
                }
                
                # If the impersonation gate score is less than the threshold, short-circuit to Safe!
                if s1_prob < impersonation_threshold:
                    _domain = features['sender_email'].split('@')[1] if '@' in features['sender_email'] else ''
                    _explanation = build_explanation(
                        prediction='Safe',
                        model_used='sec19_two_stage_header_gate',
                        s1_prob=s1_prob,
                        sender_email=features['sender_email'],
                        sender_name=features['sender_name'],
                        reply_to=features['reply_to_email'],
                        domain=_domain,
                        name_matches_email=(features['name_email_mismatch'] == 0),
                        reply_to_mismatch=features['reply_to_mismatch'],
                        sender_rarity=features['sender_rarity'],
                        signature_mismatch=False,
                        fallback_used=False
                    )
                    return jsonify({
                        'prediction': 'Safe',
                        'confidence': float(1.0 - s1_prob),
                        'original_text_length': len(text_to_test),
                        'model_used': 'sec19_two_stage_header_gate',
                        'fallback_used': False,
                        'sender_info': sender_info,
                        'enforced_sender_mismatch': False,
                        'impersonation_score': s1_prob,
                        'explanation': _explanation
                    })
                
                # Else, proceed to Stage 2: Content Classifier
                header_part, body_part = extract_header_body(full_text)
                text_for_ml = body_part if len(body_part.strip()) > 10 else full_text
                
                # Preprocess body text exactly as in baseline training
                cleaned_body = preprocess_body_text(text_for_ml)
                
                # TF-IDF
                body_features = sec19_body_vec.transform([cleaned_body])
                
                # Stage 2 KNN prob
                s2_prob = float(sec19_body_model.predict_proba(body_features)[0][1])
                
                # Combine using stacking Logistic Regression
                stack_features = np.array([[s1_prob, s2_prob]])
                final_prob = float(sec19_final_model.predict_proba(stack_features)[0][1])
                final_pred = int(sec19_final_model.predict(stack_features)[0])
                
                result = "Phishing" if final_prob >= 0.50 else "Safe"
                confidence = final_prob if result == "Phishing" else (1.0 - final_prob)
                
                _domain = features['sender_email'].split('@')[1] if '@' in features['sender_email'] else ''
                _explanation = build_explanation(
                    prediction=result,
                    model_used='sec19_two_stage',
                    s1_prob=s1_prob,
                    s2_prob=s2_prob,
                    final_prob=final_prob,
                    sender_email=features['sender_email'],
                    sender_name=features['sender_name'],
                    reply_to=features['reply_to_email'],
                    domain=_domain,
                    name_matches_email=(features['name_email_mismatch'] == 0),
                    reply_to_mismatch=features['reply_to_mismatch'],
                    sender_rarity=features['sender_rarity'],
                    signature_mismatch=sender_info['suspicious_sender_mismatch'],
                    fallback_used=False
                )
                return jsonify({
                    'prediction': result,
                    'confidence': float(confidence),
                    'original_text_length': len(text_to_test),
                    'model_used': 'sec19_two_stage',
                    'fallback_used': False,
                    'sender_info': sender_info,
                    'enforced_sender_mismatch': sender_info['suspicious_sender_mismatch'],
                    'impersonation_score': s1_prob,
                    'stage_2_ml_score': s2_prob,
                    'final_probability': final_prob,
                    'explanation': _explanation
                })
            except Exception as e:
                print(f"USENIX '19 cascade prediction failed: {e}")
                ml_score = fallback_phishing_score(text_to_test)
                result = "Phishing" if ml_score >= 0.60 else "Safe"
                return jsonify({
                    'prediction': result,
                    'confidence': float(ml_score),
                    'original_text_length': len(text_to_test),
                    'model_used': 'sec19_two_stage_fallback',
                    'fallback_used': True,
                    'sender_info': {
                        'header_sender': '',
                        'header_email': '',
                        'signature_name': '',
                        'name_matches_email': False,
                        'name_matches_signature': False,
                        'suspicious_sender_mismatch': True
                    },
                    'enforced_sender_mismatch': True,
                    'impersonation_score': 0.5,
                    'stage_2_ml_score': ml_score
                })
        else:
            ml_score = fallback_phishing_score(text_to_test)
            result = "Phishing" if ml_score >= 0.60 else "Safe"
            return jsonify({
                'prediction': result,
                'confidence': float(ml_score),
                'original_text_length': len(text_to_test),
                'model_used': 'sec19_two_stage_not_loaded_fallback',
                'fallback_used': True,
                'sender_info': {
                    'header_sender': '',
                    'header_email': '',
                    'signature_name': '',
                    'name_matches_email': False,
                    'name_matches_signature': False,
                    'suspicious_sender_mismatch': True
                },
                'enforced_sender_mismatch': True,
                'impersonation_score': 0.5,
                'stage_2_ml_score': ml_score
            })

    # ---------------------------------------------------------
    # STAGE 1: IMPERSONATION GATE (HEADER ANALYSIS)
    # ---------------------------------------------------------
    full_text = str(text_to_test)
    
    # 1. Parse Header
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

    # 2. Extract Stage 1 Features
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

    # 3. Calculate Stage 1 Impersonation Risk Score
    impersonation_score = 0.0
    
    # Feature A: Free email domain vs corporate domain
    domain = header_email.split('@')[1].lower() if '@' in header_email else ""
    common_providers = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}
    if domain in common_providers:
        impersonation_score += 0.4  # Moderate risk for executives using freemail

    # Feature B: Display Name vs Email Name Mismatch
    if header_name_norm and email_local_norm and not name_matches_email:
        impersonation_score += 0.5  # High risk

    # Feature C: Signature Mismatch
    if (signature_name and header_email and not signature_matches_email):
        sender_info['suspicious_sender_mismatch'] = True
        impersonation_score += 0.6  # High risk

    # Stage 1 Decision: The Gate
    impersonation_threshold = 0.5
    
    # If the email looks perfectly legitimate from the header, short-circuit and return Safe
    # This prevents the body classifier from throwing false positives on legitimate emails
    if impersonation_score < impersonation_threshold:
        _reply_to_mismatch_flag = 0  # legacy gate doesn't have reply_to_mismatch directly
        _legacy_expl = build_explanation(
            prediction='Safe',
            model_used='stage_1_header_gate',
            impersonation_score=impersonation_score,
            sender_email=header_email,
            sender_name=header_sender_raw,
            domain=domain,
            name_matches_email=name_matches_email,
            reply_to_mismatch=0,
            signature_mismatch=sender_info.get('suspicious_sender_mismatch', False),
            fallback_used=False
        )
        return jsonify({
            'prediction': 'Safe',
            'confidence': max(0.0, 1.0 - impersonation_score),
            'original_text_length': len(text_to_test),
            'model_used': 'stage_1_header_gate',
            'fallback_used': False,
            'sender_info': sender_info,
            'enforced_sender_mismatch': False,
            'impersonation_score': impersonation_score,
            'explanation': _legacy_expl
        })


    # ---------------------------------------------------------
    # STAGE 2: CONTENT CLASSIFIER (BODY ANALYSIS)
    # ---------------------------------------------------------
    # Only execute this heavy processing if Stage 1 flagged the email
    
    # Extract body by removing header lines for cleaner NLP processing
    header_part, body_part = extract_header_body(full_text)
    text_for_ml = body_part if len(body_part.strip()) > 10 else full_text

    if model_choice in ('rf', 'random_forest', 'randomforest'):
        selected_model = 'random_forest'
        if rf_model is not None and rf_vectorizer is not None:
            try:
                features = rf_vectorizer.transform([str(text_for_ml)])
                if hasattr(rf_model, 'predict_proba'):
                    ml_score = float(rf_model.predict_proba(features)[0][1])
                else:
                    pred = int(rf_model.predict(features)[0])
                    ml_score = float(pred)
                result = "Phishing" if ml_score >= 0.60 else "Safe"
                confidence = ml_score
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
                enc = tokenizer([str(text_for_ml)], max_length=192, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                with torch.no_grad():
                    logits, mp, anom = model(input_ids, attention_mask)
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                ml_score = float(probs[0]) if hasattr(probs, '__len__') else float(probs)
                result = "Phishing" if ml_score >= 0.60 else "Safe"
                confidence = ml_score
                model_fallback = False
            except Exception as e:
                print(f"Distil model inference failed: {e}")
                model_fallback = True
        else:
            model_fallback = True

    if model_fallback:
        ml_score = fallback_phishing_score(text_for_ml)
        result = "Phishing" if ml_score >= 0.60 else "Safe"
        confidence = float(ml_score)

    # Final Stage 2 Decision: Combine header suspicion with body suspicion
    # If header is highly suspicious AND body has even slight suspicion, flag it.
    if impersonation_score >= 0.8 and ml_score >= 0.4:
        result = 'Phishing'
        confidence = max(confidence, 0.90)

    # Attach sender_info, explanation and scores to response
    _expl = build_explanation(
        prediction=result,
        model_used=selected_model,
        ml_score=ml_score,
        impersonation_score=impersonation_score,
        sender_email=header_email,
        sender_name=header_sender_raw,
        domain=domain,
        name_matches_email=name_matches_email,
        reply_to_mismatch=0,  # legacy gate uses score-based features, not binary flag
        signature_mismatch=sender_info.get('suspicious_sender_mismatch', False),
        fallback_used=model_fallback
    )
    resp = {
        'prediction': result,
        'confidence': float(confidence),
        'original_text_length': len(text_to_test),
        'model_used': selected_model,
        'fallback_used': model_fallback,
        'sender_info': sender_info,
        'enforced_sender_mismatch': sender_info.get('suspicious_sender_mismatch', False),
        'impersonation_score': impersonation_score,
        'stage_2_ml_score': ml_score,
        'explanation': _expl
    }
    return jsonify(resp)
if __name__ == '__main__':
    app.run(debug=True, port=5000)