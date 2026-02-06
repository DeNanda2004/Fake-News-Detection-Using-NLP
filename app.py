import streamlit as st
import joblib
import os
import sys

# Add src to path to import preprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import preprocessing

# Load Model
MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f5f7fa;
    }
    /* .stTextArea textarea {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    } */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    h1 {
        color: #1E3A8A;
        text-align: center;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .real {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detection System")
st.markdown("Enter a news headline or article excerpt below to check if it's likely **Real** or **Fake**.")

news_text = st.text_area("News Content", height=150, placeholder="Paste news text here...")

if st.button("Analyze News"):
    if not news_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        model = load_model()
        if model:
            # Preprocess
            with st.spinner('Analyzing...'):
                processed_text = preprocessing.preprocess_text(news_text, method='stemming')
                
                # Predict
                prediction = model.predict([processed_text])[0]
                probabilities = model.predict_proba([processed_text])[0]
                
                # Get confidence
                class_labels = model.classes_
                if prediction == 'REAL':
                    confidence = probabilities[list(class_labels).index('REAL')]
                    result_class = 'real'
                    icon = "✅"
                else:
                    confidence = probabilities[list(class_labels).index('FAKE')]
                    result_class = 'fake'
                    icon = "⚠️"
            
            # Display Result
            st.markdown(f"""
            <div class="result-card {result_class}">
                <h2>{icon} Predicted: {prediction}</h2>
                <p>Confidence: {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Debug/Explain (optional)
            with st.expander("See processed text"):
                st.write(processed_text)
                
        else:
            st.error("Model not found! Please train the model first by running `src/train.py`.")

st.markdown("---")
st.caption("Powered by Machine Learning & NLP | Created for AI Technology Project")
