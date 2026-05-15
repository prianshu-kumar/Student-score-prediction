import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("../model/model.pkl","rb"))

st.title("Student Score Predictor")

Hours = st.number_input("Hours Studies")
Pv_Score = st.number_input("Previous Score")

if st.button("Predict Score"):
    prediction = model.predict([[Pv_Score,Hours]])

    st.success(f"Predicted Score: {prediction[0].item():.2f}")