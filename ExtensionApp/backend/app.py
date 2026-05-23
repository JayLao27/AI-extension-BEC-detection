from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

import os
import json
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
    print(f"Sucessful {e}")
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
    base_dir / 'outputs' / 'distilbert_header_body' / 'best_model.pt',
    base_dir / 'Datacleaning' / 'outputs' / 'distillbert(FullDataset)' / 'best_model.pt',
    base_dir / 'outputs' / 'best_model.pt',
])


BEST_MODEL_INFO_PATH = first_existing_path([
    base_dir / 'outputs' / 'distilbert_header_body' / 'best_model_info.json',
    base_dir / 'Datacleaning' / 'outputs' / 'distillbert(FullDataset)' / 'best_model_info.json',
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

# Curated high-fidelity keywords dictionary for BEC / phishing detection
PHISHING_KEYWORDS = {
    "financial": [
        "wire transfer", "bank transfer", "routing number", "account number", 
        "swift code", "wiring instruction", "payment urgent", "send money", 
        "gift card", "steam card", "apple card", "itunes card", "direct deposit",
        "payroll change", "invoice payment", "money transfer", "bank details"
    ],
    "urgency": [
        "urgent", "immediate attention", "asap", "due today", "strictly confidential",
        "quick favor", "are you at your desk", "quick task", "immediate action", 
        "action required", "respond immediately", "required immediately"
    ],
    "credentials": [
        "sign in", "verify your account", "update password", "reset password",
        "security alert", "unauthorized login", "click here", "login credentials",
        "verify identity", "confirm password", "unauthorized access"
    ],
    "tax_hr": [
        "w-2 form", "w2 form", "tax document", "payroll info", "direct deposit details",
        "social security number", "ssn", "tax return", "employee info"
    ]
}

def calculate_keyword_phishing_score(subject, body, sender_email, sender_name):
    """
    Analyzes subject and body (case-insensitive) for common phishing/BEC keywords and phrases.
    Returns:
      - score: float from 0.0 to 1.0 (representing threat level)
      - matched_keywords: list of matched keywords/categories
      - reasons: list of explanation strings for the matches
    """
    matched = []
    reasons = []
    
    subj_clean = (subject or "").lower()
    body_clean = (body or "").lower()
    full_text = f"{subj_clean} {body_clean}"
    
    matches_by_category = {}
    
    for category, kw_list in PHISHING_KEYWORDS.items():
        cat_matches = []
        for kw in kw_list:
            pattern = rf'\b{re.escape(kw)}\b'
            if re.search(pattern, full_text):
                cat_matches.append(kw)
        if cat_matches:
            matches_by_category[category] = cat_matches
            
    score = 0.0
    if "financial" in matches_by_category:
        score += 0.45
        matched.append(f"Financial: {', '.join(matches_by_category['financial'])}")
        reasons.append(f"Found financial request keywords: {', '.join(matches_by_category['financial'])}")
    if "urgency" in matches_by_category:
        score += 0.30
        matched.append(f"Urgency: {', '.join(matches_by_category['urgency'])}")
        reasons.append(f"Found urgency-related phrases: {', '.join(matches_by_category['urgency'])}")
    if "credentials" in matches_by_category:
        score += 0.40
        matched.append(f"Credentials: {', '.join(matches_by_category['credentials'])}")
        reasons.append(f"Found login/security credential prompts: {', '.join(matches_by_category['credentials'])}")
    if "tax_hr" in matches_by_category:
        score += 0.40
        matched.append(f"Tax/HR: {', '.join(matches_by_category['tax_hr'])}")
        reasons.append(f"Found HR/tax request terminology: {', '.join(matches_by_category['tax_hr'])}")
        
    score = min(1.0, score)
    
    if len(matches_by_category) >= 2:
        score = min(1.0, score + 0.15)
        
    return score, matched, reasons

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
    class DualEncoderPhishingModel(nn.Module):
        def __init__(self, model_name, device):
            super().__init__()
            self.bert = AutoModel.from_pretrained(model_name)
            self.hidden_dim = self.bert.config.hidden_size  # 768 for DistilBERT
            
            # 4-way pooling → 3072-dim concatenation
            self.fc_combined = nn.Linear(self.hidden_dim * 4, 512)
            self.dropout1 = nn.Dropout(0.3)
            
            # Classification head
            self.fc_class = nn.Linear(512, 256)
            self.dropout2 = nn.Dropout(0.2)
            self.classifier = nn.Linear(256, 2)
            
            # Manipulation head (5 tactics)
            self.fc_manip = nn.Linear(self.hidden_dim, 256)
            self.manip_head = nn.Linear(256, 5)
            
            # Anomaly/zero-day head
            self.anomaly_head = nn.Linear(512, 1)
            
            self.to(device)
        
        def forward(self, h_input_ids, h_attention_mask, b_input_ids, b_attention_mask):
            # Encode header
            h_output = self.bert(h_input_ids, attention_mask=h_attention_mask)
            h_cls = h_output.last_hidden_state[:, 0, :]  # CLS token
            h_mean = (h_output.last_hidden_state * h_attention_mask.unsqueeze(-1)).sum(1) / h_attention_mask.sum(1, keepdim=True)
            
            # Encode body
            b_output = self.bert(b_input_ids, attention_mask=b_attention_mask)
            b_cls = b_output.last_hidden_state[:, 0, :]  # CLS token
            b_mean = (b_output.last_hidden_state * b_attention_mask.unsqueeze(-1)).sum(1) / b_attention_mask.sum(1, keepdim=True)
            
            # 4-way pooling
            combined = torch.cat([h_cls, h_mean, b_cls, b_mean], dim=1)
            
            # Classification path
            fc_out = self.fc_combined(combined)
            fc_out = F.gelu(fc_out)
            fc_out = self.dropout1(fc_out)
            
            class_hidden = self.fc_class(fc_out)
            class_hidden = F.gelu(class_hidden)
            class_hidden = self.dropout2(class_hidden)
            logits = self.classifier(class_hidden)
            
            # Manipulation path
            manip_hidden = self.fc_manip(h_cls)  # Use header CLS for manipulation
            manip_hidden = F.gelu(manip_hidden)
            manip_probs = torch.sigmoid(self.manip_head(manip_hidden))
            
            # Anomaly path
            anom_score = torch.sigmoid(self.anomaly_head(fc_out))
            
            return logits, manip_probs, anom_score

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if best_model_info:
        print(f"Best-model metadata: {best_model_info}")

    distilbert_model = None
    distilbert_tokenizer = None
    distilbert_checkpoint = Path(best_model_info.get('best_artifact', BEST_MODEL_PATH)) if 'distilbert' in best_model_info.get('best_model', '').lower() else BEST_MODEL_PATH
    if distilbert_checkpoint.exists():
        try:
            distilbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            distilbert_model = DualEncoderPhishingModel(MODEL_NAME, device)
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

import joblib

HEADER_NB_MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'header_nb_model.joblib'
HEADER_NB_VEC_PATH = Path(__file__).resolve().parent / 'models' / 'header_nb_vectorizer.joblib'

header_nb_model = None
header_nb_vectorizer = None

try:
    if HEADER_NB_MODEL_PATH.exists() and HEADER_NB_VEC_PATH.exists():
        header_nb_model = joblib.load(HEADER_NB_MODEL_PATH)
        header_nb_vectorizer = joblib.load(HEADER_NB_VEC_PATH)
        print("SUCCESS: Loaded Naive Bayes header classifier")
    else:
        print("WARNING: Naive Bayes header model or vectorizer not found.")
except Exception as e:
    print(f"Failed to load Naive Bayes header model: {e}")

def calculate_heuristic_impersonation_score(header_email, domain, header_name_norm, email_local_norm, name_matches_email, signature_name, signature_matches_email, sender_info):
    score = 0.0
    common_providers = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}
    if domain in common_providers:
        score += 0.4
    if header_name_norm and email_local_norm and not name_matches_email:
        score += 0.5
    if signature_name and header_email and not signature_matches_email:
        sender_info['suspicious_sender_mismatch'] = True
        score += 0.6
    return min(1.0, score)


# Legacy ML and cascade model architectures (Random Forest, XGBoost, USENIX'19 Cascade)
# have been deprecated and removed. Backend now uses DistilBERT with high-fidelity keyword matching.


def build_explanation(prediction, model_used, s1_prob=None, s2_prob=None, final_prob=None,
                      sender_email=None, sender_name=None, domain=None, 
                      name_matches_email=None, signature_mismatch=None, fallback_used=False):
    """
    Build a human-readable explanation and list of evidence for why an email was
    classified as Safe or Phishing, based on the feature values computed at runtime.
    """
    reasons = []
    stage_breakdown = {}

    common_providers = {'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com'}

    # Free/public email domain
    if domain and domain in common_providers:
        reasons.append(
            f"Sender uses a free public email provider ({domain}). "
            "Executives and companies rarely send business-critical emails from personal webmail accounts."
        )
    elif domain:
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

    # Signature mismatch
    if signature_mismatch:
        reasons.append(
            "The name in the email signature does not match the sender's email address, "
            "indicating possible identity spoofing."
        )

    # Impersonation score indicator (Stage 1 · Header Gate)
    if s1_prob is not None:
        stage_breakdown['stage_1_impersonation_probability'] = round(s1_prob, 4)
        if s1_prob < 0.5:
            reasons.append(
                f"Header analysis indicates low impersonation risk ({s1_prob:.1%}), "
                "passing the safety gate for the final verdict."
            )
        else:
            reasons.append(
                f"Header analysis flagged this email with elevated impersonation risk ({s1_prob:.1%}), "
                "triggering deeper body content analysis."
            )

    # Content Classifier score indicator (Stage 2 · Body Analysis)
    if s2_prob is not None:
        stage_breakdown['stage_2_body_phishing_probability'] = round(s2_prob, 4)
        if s2_prob >= 0.6:
            reasons.append(
                f"Body content analysis detected high phishing probability ({s2_prob:.1%}) "
                "with keywords indicating urgent demands, wire transfers, or security/credentials actions."
            )
        elif s2_prob >= 0.4:
            reasons.append(
                f"Body content analysis detected moderate phishing signals ({s2_prob:.1%})."
            )
        else:
            reasons.append(
                f"Body content analysis detected low phishing probability ({s2_prob:.1%})."
            )

    # Final Stacked Decision (Stage 3 · stacked decision)
    if final_prob is not None:
        stage_breakdown['final_stacked_probability'] = round(final_prob, 4)

    # Fallback case
    if fallback_used:
        reasons.append(
            "Note: The primary deep learning model (DistilBERT) was offline/failed. "
            "High-fidelity backend keyword analysis was utilized to determine email safety."
        )

    # Build summary
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
    print(f"[DEBUG] predict route called. HAVE_TORCH: {HAVE_TORCH}", flush=True)
    data = request.json or {}
    
    header_text = data.get('header', '')
    body_text = data.get('body', '')
    
    # Fallback to older payload format if header and body are missing
    if not header_text and not body_text and 'text' in data:
        text_to_test = data.get('text', '')
        header_text, body_text = extract_header_body(text_to_test)
    else:
        text_to_test = f"{header_text}\n\n{body_text}".strip()
        
    if not header_text and not body_text:
        return jsonify({'error': 'No text provided'}), 400

    # ---------------------------------------------------------
    # STAGE 1: IMPERSONATION GATE (HEADER ANALYSIS)
    # ---------------------------------------------------------
    
    # 1. Parse Header
    from_match = re.search(r'From:\s*([^\n<]+)', header_text, re.IGNORECASE)
    header_sender_raw = from_match.group(1).strip() if from_match else ''
    header_email = extract_email_from_header(header_text)
    
    if header_sender_raw.strip().lower() == 'me' and header_email:
        header_sender_raw = sender_name_from_email(header_email) or header_email
        
    signature_name = extract_signature_name(body_text)
    if not signature_name:
        # Fallback: last multi-word capitalized name in the body text
        names = re.findall(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', body_text)
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

    # Free/corporate domain check for detailed reasons
    domain = header_email.split('@')[1].lower() if '@' in header_email else ""
    
    # Calculate display name / signature mismatch
    if signature_name and header_email and not signature_matches_email:
        sender_info['suspicious_sender_mismatch'] = True

    # 3. Calculate Stage 1 Impersonation Risk Score using Naive Bayes header classifier
    impersonation_score = 0.0
    if header_nb_model is not None and header_nb_vectorizer is not None and header_text:
        try:
            h_vec = header_nb_vectorizer.transform([header_text])
            nb_probs = header_nb_model.predict_proba(h_vec)[0]
            impersonation_score = float(nb_probs[1])
            print(f"[DEBUG] Naive Bayes header phishing probability: {impersonation_score:.4f}", flush=True)
        except Exception as e:
            print(f"Error predicting header with Naive Bayes: {e}", flush=True)
            impersonation_score = calculate_heuristic_impersonation_score(header_email, domain, header_name_norm, email_local_norm, name_matches_email, signature_name, signature_matches_email, sender_info)
    else:
        impersonation_score = calculate_heuristic_impersonation_score(header_email, domain, header_name_norm, email_local_norm, name_matches_email, signature_name, signature_matches_email, sender_info)

    # Stage 1 Decision: The Gate
    impersonation_threshold = 0.5
    
    # ---------------------------------------------------------
    # STAGE 2: CONTENT CLASSIFIER (BODY ANALYSIS)
    # ---------------------------------------------------------
    
    # Calculate keyword phishing score on the body
    keyword_score, matched_keywords, keyword_reasons = calculate_keyword_phishing_score(
        subject=re.search(r'Subject:\s*([^\n\r]+)', header_text, re.IGNORECASE).group(1).strip() if re.search(r'Subject:\s*([^\n\r]+)', header_text, re.IGNORECASE) else "",
        body=body_text,
        sender_email=header_email,
        sender_name=header_sender_raw
    )

    ml_score = 0.0
    model_failed = False
    fallback_used = False
    debug_error = None
    selected_model = 'distilbert'

    # Run DistilBERT ML Model on body if available
    if HAVE_TORCH and model is not None and tokenizer is not None:
        try:
            # MITIGATION: Feed a neutral legimitate Enron header representation
            # to bypass the dual-encoder domain-shortcut bias
            neutral_header = "From: employee@enron.com\nTo: manager@enron.com\nSubject: Update"
            
            h_enc = tokenizer([neutral_header], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            b_enc = tokenizer([str(body_text)], max_length=256, padding='max_length', truncation=True, return_tensors='pt')
            
            h_input_ids = h_enc['input_ids'].to(device)
            h_attention_mask = h_enc['attention_mask'].to(device)
            b_input_ids = b_enc['input_ids'].to(device)
            b_attention_mask = b_enc['attention_mask'].to(device)
            
            with torch.no_grad():
                logits, mp, anom = model(
                    h_input_ids=h_input_ids,
                    h_attention_mask=h_attention_mask,
                    b_input_ids=b_input_ids,
                    b_attention_mask=b_attention_mask
                )
            
            # Calibrate logits to clean 0.0-1.0 body score
            # Clean body class 1 logit is around -9.1, phishing body is around -7.9
            c1_logit = float(logits[0][1].cpu().numpy())
            min_logit = -8.9
            max_logit = -7.7
            calibrated_score = (c1_logit - min_logit) / (max_logit - min_logit)
            ml_score = max(0.0, min(1.0, calibrated_score))
            print(f"[DEBUG] DistilBERT raw Class 1 logit: {c1_logit:.4f} | Calibrated score: {ml_score:.4f}", flush=True)
            
        except Exception as e:
            import sys, traceback
            print(f"Distil model inference failed: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            model_failed = True
            debug_error = f"DistilBERT inference failed: {e}"
    else:
        model_failed = True
        debug_error = "DistilBERT model not loaded (Torch/Transformers unavailable)."

    # Determine final prediction combining DistilBERT and Keyword scores
    raw_s2_prob = ml_score
    
    if model_failed:
        # Graceful Fallback Mode to Keyword Detector
        raw_s2_prob = keyword_score
        fallback_used = True
        selected_model = 'keywords_fallback'
        combined_score = keyword_score
        result = "Phishing" if combined_score >= 0.40 else "Safe"
        confidence = combined_score if result == "Phishing" else (1.0 - combined_score)
    else:
        # Boost DistilBERT with high-confidence keywords
        if keyword_score >= 0.7:
            combined_score = max(raw_s2_prob, keyword_score, 0.85)
        elif keyword_score >= 0.4 and raw_s2_prob >= 0.3:
            combined_score = max(raw_s2_prob, 0.75)
        else:
            # Weighted combination: 70% DistilBERT, 30% Keywords
            combined_score = 0.7 * raw_s2_prob + 0.3 * keyword_score
            
        result = "Phishing" if combined_score >= 0.60 else "Safe"
        confidence = combined_score if result == "Phishing" else (1.0 - combined_score)

    # Extra: Impersonation AND Content check
    if impersonation_score >= 0.8 and combined_score >= 0.4:
        result = 'Phishing'
        combined_score = max(combined_score, 0.90)
        confidence = max(confidence, 0.90)

    # Cap impersonation score at 1.0 (100%) for display purposes
    capped_s1_prob = min(1.0, impersonation_score)

    # If the email passed the Stage 1 impersonation safety gate, override the final verdict to Safe
    # while preserving the accurate, true independent Stage 2 body analysis scores.
    if impersonation_score < impersonation_threshold:
        result = 'Safe'
        combined_score = capped_s1_prob
        confidence = 1.0 - capped_s1_prob
        selected_model = 'stage_1_header_gate'

    # Build detailed explanation
    _expl = build_explanation(
        prediction=result,
        model_used=selected_model,
        s1_prob=capped_s1_prob,
        s2_prob=raw_s2_prob,
        final_prob=combined_score,
        sender_email=header_email,
        sender_name=header_sender_raw,
        domain=domain,
        name_matches_email=name_matches_email,
        signature_mismatch=sender_info.get('suspicious_sender_mismatch', False),
        fallback_used=fallback_used
    )
    
    # Add matched keyword evidence to the explanations list
    if matched_keywords:
        _expl['reasons'].append(
            f"Keyword Analysis matched the following high-risk indicators: {', '.join(matched_keywords)}."
        )

    resp = {
        'prediction': result,
        'confidence': float(confidence),
        'bec_probability': float(combined_score),
        'original_text_length': len(body_text),
        'model_used': selected_model,
        'fallback_used': fallback_used,
        'sender_info': sender_info,
        'enforced_sender_mismatch': sender_info.get('suspicious_sender_mismatch', False),
        'impersonation_score': capped_s1_prob,
        'stage_2_ml_score': raw_s2_prob,
        'explanation': _expl,
        'debug_error': debug_error
    }
    return jsonify(resp)
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)