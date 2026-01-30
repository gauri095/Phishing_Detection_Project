import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


# ------------------------------------------------------------------
# URL MODEL
# ------------------------------------------------------------------

def ensure_url_model(path: str):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Demo numeric model (placeholder)
    X = np.zeros((20, 10))
    X[:10, 0] = 1
    y = np.array([1] * 10 + [0] * 10)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)

    feature_columns = [
        "url_length",
        "host_length",
        "path_length",
        "has_ip",
        "num_subdomains",
        "num_dots",
        "num_hyphens",
        "has_at",
        "has_query",
        "num_query_params",
    ]

    joblib.dump(
        {
            "model": clf,
            "feature_columns": feature_columns,
        },
        path,
    )


# ------------------------------------------------------------------
# TEXT MODEL
# ------------------------------------------------------------------

def ensure_text_model(path: str):
    if os.path.exists(path):
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)

    phishing_examples = [
        "Your account has been suspended. Click here to verify: http://fake-login.example/login",
        "We noticed suspicious activity. Please confirm your password immediately.",
        "Urgent: Your bank account will be closed. Verify now.",
    ]

    legit_examples = [
        "Hi team, the meeting is scheduled for 10am tomorrow. Thanks!",
        "Your order #12345 has shipped and is on the way.",
    ]

    X = phishing_examples + legit_examples
    y = np.array([1] * len(phishing_examples) + [0] * len(legit_examples))

    pipe = make_pipeline(
        TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            stop_words="english",
        ),
        LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42,
        ),
    )

    pipe.fit(X, y)
    joblib.dump({"text_model": pipe}, path)


# ------------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------------

def ensure_models(
    url_model_path="model/phish_detector.joblib",
    text_model_path="model/text_model.joblib",
):
    ensure_url_model(url_model_path)
    ensure_text_model(text_model_path)


def load_url_model_bundle(path: str):
    return joblib.load(path)


def load_text_model_bundle(path: str):
    return joblib.load(path)


# ------------------------------------------------------------------
# PREDICTION — URL
# ------------------------------------------------------------------

def predict_url_bundle(bundle, df, explain=False):
    """
    df must contain numeric URL features.
    Extra columns are ignored safely.
    Missing columns are filled with 0.
    """

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    # Drop non-numeric / non-feature columns
    df_numeric = df.copy()

    if "registered_domain" in df_numeric.columns:
        df_numeric = df_numeric.drop(columns=["registered_domain"])

    # Ensure all expected features exist
    for col in feature_columns:
        if col not in df_numeric.columns:
            df_numeric[col] = 0

    # Select & order columns EXACTLY as during training
    X = df_numeric[feature_columns].fillna(0)

    proba = float(model.predict_proba(X)[:, 1][0])
    label = "phishing" if proba >= 0.5 else "legitimate"

    explanation = None
    return proba, label, explanation


# ------------------------------------------------------------------
# PREDICTION — TEXT
# ------------------------------------------------------------------

def predict_text_bundle(bundle, message: str, explain: bool = True):
    pipe = (
        bundle["text_model"]
        if isinstance(bundle, dict) and "text_model" in bundle
        else bundle
    )

    proba = float(pipe.predict_proba([message])[:, 1][0])
    label = "phishing" if proba >= 0.5 else "legitimate"

    contributions = None

    if explain:
        try:
            vect = pipe.named_steps["tfidfvectorizer"]
            lr = pipe.named_steps["logisticregression"]

            features = vect.get_feature_names_out()
            coefs = lr.coef_[0]

            msg_vec = vect.transform([message])
            present_idx = msg_vec.nonzero()[1]

            token_scores = [
                (features[idx], float(coefs[idx])) for idx in present_idx
            ]

            token_scores.sort(key=lambda x: -x[1])
            contributions = pd.DataFrame(
                token_scores[:20],
                columns=["token", "coef"],
            )

        except Exception as e:
            contributions = pd.DataFrame([{"error": str(e)}])

    return proba, label, contributions
