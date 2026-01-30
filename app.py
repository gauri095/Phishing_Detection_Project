import streamlit as st
import pandas as pd
import joblib
import os
import random
from datetime import datetime

from features import extract_features, extract_text_features
from model_utils import (
    ensure_models,
    load_url_model_bundle,
    load_text_model_bundle,
    predict_url_bundle,
    predict_text_bundle,
)
# AI transformer inference wrapper
from text_inference import TextTransformerClassifier

st.set_page_config(page_title="AI Phishing Detector", layout="centered")
st.title("AI-Powered Phishing Detector (Prototype)")

# ensure fallback demo models exist
os.makedirs("model", exist_ok=True)
ensure_models(url_model_path="model/phish_detector.joblib", text_model_path="model/text_model.joblib")

# Try to load a fine-tuned transformer model if available
TEXT_TRANSFORMER_DIR = "model/text_distilbert"
text_transformer = None
if os.path.isdir(TEXT_TRANSFORMER_DIR):
    try:
        text_transformer = TextTransformerClassifier(TEXT_TRANSFORMER_DIR)
        st.info("Loaded Transformer text model for AI-powered message classification.")
    except Exception as e:
        st.warning(f"Failed to load transformer model from {TEXT_TRANSFORMER_DIR}: {e}")
        text_transformer = None

mode = st.radio("Mode", ["URL", "Message", "Self-Test"])

# --------------------------
# URL analysis UI
# --------------------------
if mode == "URL":
    st.subheader("URL analysis")
    url = st.text_input("Enter a URL", value="http://example.com")
    explain = st.checkbox("Show explanation (if available)", value=False)
    if st.button("Predict URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            bundle = load_url_model_bundle("model/phish_detector.joblib")
            feats = extract_features(url)
            df = pd.DataFrame([feats])
            proba, label, explanation = predict_url_bundle(bundle, df, explain=explain)
            st.metric("Phishing probability", f"{proba:.3f}")
            st.write("Predicted label:", label)
            with st.expander("Features"):
                st.write(df.T)
            if explanation:
                with st.expander("Explanation"):
                    st.write(explanation)

# --------------------------
# Message analysis UI
# --------------------------
elif mode == "Message":
    st.subheader("Message analysis (email / SMS / chat)")
    message = st.text_area("Paste message text (subject + body)", height=200)
    explain_tokens = st.checkbox("Show token contributions (LIME if transformer available)", value=False)
    if st.button("Analyze Message"):
        if not message.strip():
            st.warning("Please paste a message to analyze.")
        else:
            # Prefer transformer if loaded
            if text_transformer is not None:
                proba = text_transformer.predict_proba([message])[0]
                label = "phishing" if proba >= 0.5 else "legitimate"
                st.metric("Phishing probability", f"{proba:.3f}")
                st.write("Predicted label:", label)
                if explain_tokens:
                    try:
                        from lime.lime_text import LimeTextExplainer
                        explainer = LimeTextExplainer(class_names=["legit", "phish"])
                        def _predict_for_lime(texts):
                            return [[1 - p, p] for p in text_transformer.predict_proba(texts)]
                        exp = explainer.explain_instance(message, _predict_for_lime, num_features=8)
                        html = exp.as_html()
                        st.components.v1.html(html, height=350, scrolling=True)
                    except Exception as e:
                        st.warning("LIME explanation failed or not installed: " + str(e))
            else:
                # Fallback to TF-IDF logistic pipeline
                text_bundle = load_text_model_bundle("model/text_model.joblib")
                proba, label, contributions = predict_text_bundle(text_bundle, message, explain=explain_tokens)
                st.metric("Phishing probability", f"{proba:.3f}")
                st.write("Predicted label:", label)
                if contributions is not None:
                    with st.expander("Top contributing tokens"):
                        st.table(contributions)

# --------------------------
# Self-test UI
# --------------------------
else:
    st.subheader("Phishing Self-Test")
    st.markdown(
        "Take a short quiz to test your ability to spot phishing. "
        "You will be shown short simulated emails or messages. Choose the best answer and get immediate feedback."
    )

    QUESTION_BANK = [
        {
            "id": "q1",
            "prompt": (
                "From: support@secure-payments.example\n"
                "Subject: Urgent: Verify your account\n\n"
                "Dear Customer,\n\n"
                "We detected a problem with your account. Please verify your information immediately by clicking the link below:\n"
                "http://secure-payments.example.verify-account.com/login\n\n"
                "Failure to verify will result in account suspension.\n\n"
                "Sincerely,\nCustomer Support"
            ),
            "choices": ["Phishing", "Legitimate", "Not sure"],
            "answer": 0,
            "explanation": "This is phishing: the link domain is a misleading subdomain (verify-account.com) not the bank's registered domain, and there is urgent language threatening suspension."
        },
        {
            "id": "q2",
            "prompt": (
                "From: noreply@github.com\n"
                "Subject: [GitHub] New device signed in\n\n"
                "Hi,\n\n"
                "A new device signed into your account from New York. If this was you, no action is required. If you don't recognize this activity, please visit https://github.com/settings/security and review your sessions.\n\n"
                "Regards,\nGitHub Security"
            ),
            "choices": ["Phishing", "Legitimate", "Not sure"],
            "answer": 1,
            "explanation": "This looks legitimate: it references the correct github.com domain and links to a logical settings path. However, check sender address carefully in real emails."
        },
        {
            "id": "q3",
            "prompt": (
                "From: billing@paypal.example\n"
                "Subject: Invoice attached\n\n"
                "Hello,\n\n"
                "Please see the attached invoice for your payment. Open the attachment to view details.\n\nThanks."
            ),
            "choices": ["Phishing", "Legitimate", "Not sure"],
            "answer": 0,
            "explanation": "Likely phishing: unexpected attachments from payment services are a common vector. Legit providers usually include clear reference numbers and links rather than unsolicited attachments."
        },
    ]

    # session state initialization
    if "self_test_questions" not in st.session_state:
        st.session_state.self_test_questions = []
    if "self_test_index" not in st.session_state:
        st.session_state.self_test_index = 0
    if "self_test_answers" not in st.session_state:
        st.session_state.self_test_answers = {}
    if "self_test_started" not in st.session_state:
        st.session_state.self_test_started = False

    def start_quiz(shuffle=True, n_questions=3):
        bank = QUESTION_BANK.copy()
        if shuffle:
            random.shuffle(bank)
        st.session_state.self_test_questions = bank[:n_questions]
        st.session_state.self_test_index = 0
        st.session_state.self_test_answers = {}
        st.session_state.self_test_started = True

    def record_and_next(choice_idx):
        idx = st.session_state.self_test_index
        q = st.session_state.self_test_questions[idx]
        st.session_state.self_test_answers[q["id"]] = choice_idx
        st.session_state.self_test_index = idx + 1

    def finish_and_save(score, total):
        os.makedirs("results", exist_ok=True)
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "score": score,
            "total": total,
            "percent": round(100 * score / total, 2),
            "answers": st.session_state.self_test_answers
        }
        results_file = os.path.join("results", "self_test_results.csv")
        df = pd.DataFrame([row])
        if os.path.exists(results_file):
            df.to_csv(results_file, mode="a", header=False, index=False)
        else:
            df.to_csv(results_file, index=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start Self-Test"):
            start_quiz(shuffle=True, n_questions=3)
    with col2:
        if st.button("Restart / Shuffle"):
            start_quiz(shuffle=True, n_questions=3)

    if not st.session_state.self_test_started:
        st.info("Click 'Start Self-Test' to begin.")
        st.stop()

    if st.session_state.self_test_index >= len(st.session_state.self_test_questions):
        questions = st.session_state.self_test_questions
        answers = st.session_state.self_test_answers
        total = len(questions)
        correct = 0
        st.success("Completed! Review your results below.")
        for i, q in enumerate(questions):
            qid = q["id"]
            user_ans = answers.get(qid, None)
            is_correct = (user_ans == q["answer"])
            if is_correct:
                correct += 1
            st.markdown(f"**Question {i+1}:**")
            st.write(q["prompt"])
            st.write(f"Your answer: **{q['choices'][user_ans] if user_ans is not None else 'No answer'}** — "
                     f"{'✅ Correct' if is_correct else '❌ Incorrect'}")
            st.info(f"Explanation: {q['explanation']}")
            st.write("---")
        st.metric("Score", f"{correct} / {total}", delta=f"{round(100*correct/total,1)}%")
        if st.button("Save results"):
            finish_and_save(correct, total)
            st.success("Saved to results/self_test_results.csv")
        if st.button("Retake quiz"):
            start_quiz(shuffle=True, n_questions=total)
        st.stop()

    qidx = st.session_state.self_test_index
    q = st.session_state.self_test_questions[qidx]
    st.markdown(f"### Question {qidx+1} of {len(st.session_state.self_test_questions)}")
    st.write(q["prompt"])
    choice = st.radio("Select the best answer", q["choices"], key=f"choice_{q['id']}")
    if st.button("Submit Answer"):
        choice_idx = q["choices"].index(choice)
        record_and_next(choice_idx)
        st.rerun()