import streamlit as st
import pandas as pd
import joblib
from transformers import pipeline

# Load models and summarizer
st.cache_resource

def load_models():
    tone_model = joblib.load("tone_model.joblib")
    content_model = joblib.load("content_model.joblib")
    summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")
    return tone_model, content_model, summarizer

tone_model, content_model, summarizer = load_models()

# Define tone tags (must match those used in training)
used_tone_tags = [
    'Aspirational', 'Informative', 'Inclusive', 'Urgent', 'Conversational',
    'Empathetic', 'Elegant', 'Energetic', 'Lifestyle focused', 'Professional',
    'Direct', 'FOMO driven', 'Real time', 'Warm', 'Reassuring',
    'Action Oriented', 'Promotional', 'Investment Oriented', 'Emotional',
    'Exclusive', 'Exciting'
]

# Streamlit UI
st.title("📣 Campaign Tagging + AI Summary App")

with st.form("predict_form"):
    message = st.text_area("📨 Paste campaign message here:", height=200)
    campaign_type = st.text_input("Campaign Type")
    target_project = st.text_input("Target Project")
    submit = st.form_submit_button("Predict")

if submit and message and campaign_type and target_project:
    # Prepare input
    df = pd.DataFrame([{
        "Message": message,
        "Campaign Type": campaign_type,
        "Target Project": target_project
    }])

    # Make predictions
    tone_preds = tone_model.predict(df)[0]
    tone_labels = [tag for tag, val in zip(used_tone_tags, tone_preds) if val == 1]
    content_tag = content_model.predict(df)[0]

    try:
        summary = summarizer(message, max_length=40, min_length=10, do_sample=False)[0]['summary_text']
    except:
        summary = "Error generating summary."

    # Display results
    st.subheader("🎯 Predicted Tone Tags")
    st.write(", ".join(tone_labels) if tone_labels else "No tones predicted.")

    st.subheader("🏷️ Predicted Content Tag")
    st.write(content_tag)

    st.subheader("🧠 AI-Generated Summary")
    st.info(summary)

elif submit:
    st.warning("Please fill in all fields to predict.")
