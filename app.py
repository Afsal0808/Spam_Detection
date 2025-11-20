import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📩 Spam Message Detector (Naive Bayes)")

st.write("Enter any message below and check if it's Spam or Not Spam.")

message = st.text_area("Message here...")

if st.button("Predict"):
    msg_tf = vectorizer.transform([message])
    pred = model.predict(msg_tf)[0]

    if pred == 1:
        st.error("❌ SPAM Message Detected!")
    else:
        st.success("✅ NOT SPAM (Safe Message)")
