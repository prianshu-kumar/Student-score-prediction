import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("../model/model.pkl", "rb"))

# Page config
st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="📘",
)

# Title
st.title("📘 Student Score Predictor")

st.write(
    "Predict student exam score based on study hours."
)

# Input
Hours = st.slider(
    "Study Hours",
    min_value=0.0,
    max_value=12.0,
    step=0.5
)
Pv_Score = st.slider(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

# Predict button
if st.button("Predict Score"):
    prediction = model.predict([[Pv_Score,Hours]])

    st.success(f"Predicted Score: {prediction[0].item():.2f}")

# Footer
st.markdown("---")
st.caption("Built using Scikit-learn & Streamlit")

