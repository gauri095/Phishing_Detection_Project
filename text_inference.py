from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os

class TextTransformerClassifier:
    """
    Simple wrapper around a fine-tuned HuggingFace sequence classification model.
    Exposes predict_proba(list[str]) -> list[float] (probability of class 1).
    """
    def __init__(self, model_dir="model/text_distilbert", device=None):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict_proba(self, texts, batch_size=8):
        all_probs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                enc = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.detach().cpu().numpy()
                exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                all_probs.extend(probs[:, 1].tolist())
        return all_probs

    def predict(self, text):
        p = self.predict_proba([text])[0]
        label = "phishing" if p >= 0.5 else "legitimate"
        return p, label