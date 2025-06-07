# =================================================================================
# SentiMind: A High-Accuracy Sentiment Analysis Web Application
#
# Author: [Your Name]
# Date: [Current Date]
#
# Description:
# This Streamlit application provides a user-friendly interface for sentiment
# analysis. It leverages a state-of-the-art, pre-trained RoBERTa model from the
# Hugging Face Transformers library to deliver highly accurate predictions
# on user-provided text.
# =================================================================================

# --- 1. IMPORTS ---
# Import necessary libraries.
# Streamlit is for creating the web app interface.
# Transformers/PyTorch are for loading and running the AI model.
import streamlit as st
from transformers import pipeline
import torch

# --- 2. SETUP & CONFIGURATION ---
# Configure the Streamlit page. This must be the first Streamlit command.
st.set_page_config(
    page_title="SentiMind 🤖",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)


# --- 3. MODEL LOADING ---
# Use a caching decorator to prevent the model from being reloaded on every
# user interaction, which significantly improves app performance.
@st.cache_resource
def load_sentiment_model():
    """
    Loads and caches the sentiment analysis pipeline from Hugging Face.
    This function is decorated with @st.cache_resource to ensure that
    the expensive model loading process is only performed once.
    
    Returns:
        transformers.pipeline: A ready-to-use sentiment analysis pipeline.
    """
    print("Initializing sentiment analysis model...")
    # Automatically select GPU if available for faster inference, else fallback to CPU.
    device = 0 if torch.cuda.is_available() else -1
    
    # Load a RoBERTa model fine-tuned specifically on Twitter data for sentiment tasks.
    # This provides better performance on short, informal text.
    model_pipeline = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device
    )
    print("Model loaded successfully.")
    return model_pipeline

# Load the model into the app's global scope.
sentiment_classifier = load_sentiment_model()


# --- 4. USER INTERFACE ---
# Define the main function that lays out the app's UI.
def main():
    """
    Main function to define and render the Streamlit user interface.
    """
    # Set the main title and a descriptive sub-header.
    st.title("SentiMind 🤖")
    st.subheader("A precision accuracy instrument to assist you in determining the emotional level of any text.")

    # Add a divider for better visual structure.
    st.divider()

    # Create a text area for user input.
    user_input = st.text_area(
        "Enter text or a tweet to analyze:",
        height=150,
        placeholder="Example: I am thrilled to learn about deploying AI models!"
    )

    # Create a button to trigger the analysis.
    if st.button("Analyze Sentiment"):
        # Ensure the user has provided some input before proceeding.
        if user_input:
            # The pipeline returns a list containing a dictionary, e.g., [{'label': 'positive', 'score': 0.98}]
            with st.spinner("Analyzing..."):
                result = sentiment_classifier(user_input)[0]
                label = result['label']
                score = result['score']
            
            # Display the results in a structured and visually appealing way.
            display_results(label, score)
        else:
            # Display a warning if the text area is empty.
            st.warning("Please enter some text to analyze.")

def display_results(label, score):
    """
    Displays the sentiment analysis results in a formatted way.

    Args:
        label (str): The predicted sentiment label (e.g., 'positive').
        score (float): The confidence score of the prediction.
    """
    st.subheader("Analysis Result")

    # Use columns to display the sentiment and score side-by-side.
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Sentiment", label.capitalize())
    with col2:
        st.metric("Confidence", f"{score:.2%}")

    # Provide colored feedback boxes for a more intuitive user experience.
    if label == 'positive':
        st.success(f"The model is {score:.2%} confident that the sentiment is Positive.")
    elif label == 'negative':
        st.error(f"The model is {score:.2%} confident that the sentiment is Negative.")
    else:  # neutral
        st.info(f"The model is {score:.2%} confident that the sentiment is Neutral.")


# --- 5. APP EXECUTION ---
# This is the standard entry point for a Python script.
if __name__ == "__main__":
    main()