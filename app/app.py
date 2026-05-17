import streamlit as st
import pickle
import numpy as np
from pathlib import Path
import logging

# Attempt to locate the model file from several likely locations
BASE_DIR = Path(__file__).resolve().parent
candidate_paths = [
    BASE_DIR / "model" / "model.pkl",
    BASE_DIR.parent / "model" / "model.pkl",
    Path.cwd() / "model" / "model.pkl",
]

MODEL_PATH = None
for p in candidate_paths:
    if p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    logging.error("Model file not found. Checked paths:\n%s", "\n".join(str(p) for p in candidate_paths))
    st.error(
        "Model file not found. Ensure `model/model.pkl` is included in the deployed app."
    )
    st.write("Checked paths:")
    for p in candidate_paths:
        st.write(f"- {p}")
    st.stop()

with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

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

