# =================================================================================
# SentiMind: Final, Lightweight, and Deployable Version
# ==============================================================================

import streamlit as st
from transformers import pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="SentiMind 🤖",
    page_icon="🤖",
    layout="centered"
)

# --- Model Loading ---
@st.cache_resource
def load_sentiment_model():
    """
    Loads a lightweight, distilled version of the BERT model for sentiment analysis.
    This model is much smaller and faster, making it ideal for deployment.
    """
    print("Loading lightweight DistilBERT model...")
    model_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully.")
    return model_pipeline

sentiment_classifier = load_sentiment_model()

# --- App Interface ---
st.title("SentiMind 🤖")
st.subheader("A precision accuracy instrument to assist you in determining the emotional level of any text.")
st.divider()

user_input = st.text_area(
    "Enter text to analyze:",
    height=150,
    placeholder="Example: I am thrilled to learn about deploying AI models!"
)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing..."):
            result = sentiment_classifier(user_input)[0]
            label = result['label'].upper()
            score = result['score']

        st.subheader("Analysis Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", label)
        with col2:
            st.metric("Confidence", f"{score:.2%}")

        if label == 'POSITIVE':
            st.success(f"The model is {score:.2%} confident that the sentiment is Positive.")
        else: # NEGATIVE
            st.error(f"The model is {score:.2%} confident that the sentiment is Negative.")
    else:
        st.warning("Please enter some text.")
