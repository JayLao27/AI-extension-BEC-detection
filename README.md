# AI Extension for BEC Detection

## 📌 Overview

This project is an **AI-powered browser extension** designed to detect **Business Email Compromise (BEC)** and phishing attempts in real time.
It uses a machine learning pipeline to analyze email features and alert users before they fall victim to scams.

The system is optimized for:

* **High recall** → minimizing missed attacks
* **Lightweight deployment** → fast inference in-browser
* **Privacy** → runs locally using ONNX

---

## 🧠 System Architecture

The system follows a **5-phase pipeline**:

### 🔹 Phase 1: Data Collection

We gather and combine datasets from:

* CEAS-08 dataset
* Enron email dataset
* Synthetic BEC samples
* Public phishing corpora

---

### 🔹 Phase 2: Feature Engineering

Extract meaningful features from emails:

**Sender & Domain Signals**

* Sender patterns
* Domain edit distance
* Display-name spoofing

**Text Analysis**

* NLP / TF-IDF
* Urgency scoring
* Wire-transfer keywords

**Email Metadata**

* Header metadata
* SPF / DKIM flags
* Reply-To mismatch

---

### 🔹 Phase 3: Model Training (Python)

We train machine learning models using:

* XGBoost
* LightGBM

**Training setup:**

* 80/10/10 data split
* SMOTE for class imbalance
* Optimization goal: **maximize recall** (catch more attacks)

---

### 🔹 Phase 4: Export to ONNX

The trained model is converted for browser use:

* `sklearn → skl2onnx → model.onnx`
* Model size: ~200 KB

---

### 🔹 Phase 5: Extension Inference

Inside the browser extension:

* Service worker loads ONNX model
* Email is analyzed in real time
* Risk score is generated
* Warning/alert is injected into the UI

---

### 🔁 User Feedback Loop

User interactions (e.g., marking emails safe or malicious) can be used to:

* Improve future training
* Refine detection accuracy

---

## ⚙️ Technologies Used

* Python (model training)
* scikit-learn
* XGBoost / LightGBM
* ONNX / ONNX Runtime Web
* JavaScript (browser extension)

---

## 🚀 Features

* Real-time email threat detection
* Lightweight and fast inference
* Offline/local processing for privacy
* High recall to reduce missed attacks
* Easy integration into browser environments

---

## 📂 Project Structure (Example)

```
/data
/models
  └── model.onnx
/src
  ├── feature_engineering.py
  ├── train_model.py
  ├── convert_onnx.py
/extension
  ├── service_worker.js
  ├── inference.js
  └── ui/
README.md
```

---

## 🧪 Future Improvements

* Add deep learning (transformer-based NLP)
* Improve phishing URL detection
* Adaptive learning from user feedback
* Multi-language support

---

## 👥 Contributors

* Ryle Jade Tabay
* Jaymark Burlado

---

## 📜 License

This project is for academic and research purposes. License details can be added here.
