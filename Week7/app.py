import streamlit as st
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(_file_))

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("Review Sentiment Predictor")

review = st.text_area("Enter a product review:")

if st.button("Predict Sentiment"):
    vec = tfidf.transform([review])
    prediction = model.predict(vec)[0]
    st.subheader(f"Predicted Sentiment: {prediction}")
