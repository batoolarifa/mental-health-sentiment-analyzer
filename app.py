# app.py
import streamlit as st
import pickle
from utils.model_utils import SentimentModel  # your modular class

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Mental Health Sentiment Analyzer", layout="centered")
st.title("🧠 Mental Health Sentiment Analyzer")
st.markdown(
    "Type a statement about your mental health or feelings, and the model will predict its sentiment."
)

# ----------------------------
# Automatically load model and encoder
# ----------------------------
MODEL_DIR = "./saved_model"  # path to your saved model folder

try:
    # Load labels from the saved LabelEncoder
    with open(f"{MODEL_DIR}/label_encoder.pkl", "rb") as f:
        labels = pickle.load(f).classes_  # original label names

    # Initialize SentimentModel
    sentiment_model = SentimentModel(MODEL_DIR, labels)
    
except Exception as e:
   
    sentiment_model = None

st.header("Predict Mental Health Sentiment")
user_text = st.text_area("Type your statement here...")

if st.button("Predict") and user_text:
    if not sentiment_model:
        st.warning("Model not loaded properly.")
    else:
        prediction = sentiment_model.predict(user_text)
        st.success(f"Predicted Sentiment: **{prediction.upper()}**")
