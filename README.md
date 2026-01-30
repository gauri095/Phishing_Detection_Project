# Phishing_Detection_Project
# AI-Powered Phishing Detector (no stalker analyzer)

This project provides an AI-enhanced phishing detector combining URL heuristics and an AI text classifier (DistilBERT). It also includes a simple Phishing Self-Test.

Quick overview
- URL detection: engineered features + demo RandomForest (fast).
- Message detection: preferred path uses a fine-tuned DistilBERT model (directory model/text_distilbert). If not present, a small TF-IDF + LogisticRegression fallback is used (model/text_model.joblib).
- Streamlit UI: URL, Message, Self-Test.
- Training script: train_text_transformer.py (fine-tune DistilBERT).

How to run (dev)
1. Create venv & install:
   python -m venv .venv
   source .venv/bin/activate     # macOS / Linux
   .venv\\Scripts\\Activate.ps1  # Windows PowerShell
   pip install --upgrade pip
   pip install -r requirements.txt

2a. (Optional) Train the Transformer text model (requires dataset & GPU recommended):
   Prepare CSV files:
     data/messages_train.csv
     data/messages_valid.csv
   Each with columns: message,label  (label: 1 for phishing, 0 for legitimate)
   Then:
     python train_text_transformer.py --train-csv data/messages_train.csv --valid-csv data/messages_valid.csv --output-dir model/text_distilbert --epochs 3

2b. Or skip training and use the fallback TF-IDF model (auto-created on first run).

3. Start the app:
   streamlit run app.py

4. Open browser:
   http://localhost:8501

Notes & tips
- Transformer model loading can be slow on first start; prefer GPU for training and lower latency inference.
- For production, convert the HF model to ONNX and/or quantize to reduce latency and memory.
- LIME explanations are optional; install `lime` to enable token-level explanations for the transformer.
- Keep sensitive messages local; avoid uploading private data to third-party services.

If you want next
- I can provide an ONNX export script and a small FastAPI wrapper for low-latency inference.
- I can add multimodal stacking (combine URL + message model predictions).
- I can add a training pipeline to produce a production-ready dataset and evaluation metrics.
