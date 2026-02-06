import streamlit as st
import joblib
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import preprocessing

MODEL_PATH = "model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
body {
    background-color: #f6f8fc;
}

h1 {
    color: #1f2937;
    text-align: center;
}

.description {
    text-align: center;
    color: #6b7280;
    font-size: 15px;
}

.card {
    padding: 22px;
    border-radius: 12px;
    margin-top: 20px;
    text-align: center;
}

.real {
    background-color: #e8f5e9;
    color: #1b5e20;
}

.fake {
    background-color: #fdecea;
    color: #7f1d1d;
}

.info {
    background-color: #eef2ff;
    color: #3730a3;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    font-size: 14px;
}

.stButton>button {
    width: 100%;
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- APP UI ----------------
st.title("Fake News Detection System")
st.markdown(
    "<div class='description'>Analyze news content using Natural Language Processing and Machine Learning</div>",
    unsafe_allow_html=True
)

news_text = st.text_area(
    "News Content",
    height=200,
    placeholder="Paste a detailed news article or report here..."
)

if st.button("Analyze"):
    if not news_text.strip():
        st.info("Please enter news content for analysis.")
        st.stop()

    # Input length check
    if len(news_text.split()) < 20:
        st.markdown("""
        <div class="info">
        The entered text is very short.  
        For better accuracy, please provide a more detailed news article.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    model = load_model()

    with st.spinner("Processing..."):
        processed_text = preprocessing.preprocess_text(
            news_text,
            method="stemming"
        )

        prediction = model.predict([processed_text])[0]
        probabilities = model.predict_proba([processed_text])[0]
        confidence = float(np.max(probabilities))

    # Low-confidence notice (soft)
    if confidence < 0.60:
        st.markdown("""
        <div class="info">
        This article differs from the data used during training.  
        The prediction is shown with reduced confidence.
        </div>
        """, unsafe_allow_html=True)

    result_class = "real" if prediction == "REAL" else "fake"
    label = "Real News" if prediction == "REAL" else "Fake News"

    st.markdown(f"""
    <div class="card {result_class}">
        <h2>{label}</h2>
        <p>Prediction Confidence: {confidence*100:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Preprocessed Text"):
        st.write(processed_text)

st.markdown("---")
st.caption("Final-Year Project | NLP-Based Fake News Classification")
