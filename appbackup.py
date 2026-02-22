# @title 6.1 Generate Streamlit App Code

import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from bs4 import BeautifulSoup
import re

# --- CONFIG ---
MODEL_PATH = "./saved_model"
FEEDBACK_FILE = "feedback.csv"
THRESHOLD = 0.5

# --- LOAD MODEL ---
@st.cache_resource
def load_resources():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_resources()

# --- PREPROCESSING ---
def clean_text(text):
    if not text: return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# --- PREDICTION ---
def predict(name, desc):
    text = clean_text(name) + " " + clean_text(desc)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    
    # Get labels
    id2label = model.config.id2label
    results = []
    for i, prob in enumerate(probs):
        if prob > THRESHOLD:
            full_label = id2label[i]
            parts = full_label.split(" > ")
            parent = parts[0]
            sub = parts[1] if len(parts) > 1 else "None"
            results.append({"Parent": parent, "Sub": sub, "Conf": prob})
    
    return results

# --- FEEDBACK SAVING ---
def save_feedback(data):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    df = pd.DataFrame([data])
    df.to_csv(FEEDBACK_FILE, mode='a', header=not file_exists, index=False)
    return True

# --- UI ---
st.title("🛍️ AI Product Categorizer (Multi-Label)")
st.write("Enter product details to predict Parent and Sub-Categories.")

with st.form("input_form"):
    p_name = st.text_input("Product Name", "")
    p_desc = st.text_area("Description", "")
    submitted = st.form_submit_button("Predict")

if submitted:
    preds = predict(p_name, p_desc)
    
    st.subheader("Predictions")
    
    if not preds:
        st.warning("No categories detected above threshold.")
    else:
        # Store predictions in session state for feedback
        st.session_state['current_preds'] = preds
        st.session_state['p_name'] = p_name
        st.session_state['p_desc'] = p_desc
        
        for p in preds:
            st.success(f"**{p['Parent']}** > {p['Sub']} ({p['Conf']:.2%})")

# --- FEEDBACK LOOP ---
if 'current_preds' in st.session_state:
    st.divider()
    st.write("### 📝 Feedback")
    status = st.radio("Is this prediction correct?", ("Select", "Correct", "Incorrect"))
    
    if status == "Correct":
        if st.button("Save Feedback (Correct)"):
            # Save the first prediction as the primary for CSV simplicity, 
            # or save multiple rows. Here we save the top prediction.
            top_pred = st.session_state['current_preds'][0]
            row = {
                "product_name": st.session_state['p_name'],
                "product_description": st.session_state['p_desc'],
                "predicted_parent": top_pred['Parent'],
                "predicted_sub": top_pred['Sub'],
                "actual_parent": top_pred['Parent'],
                "actual_sub": top_pred['Sub'],
                "status": "Correct"
            }
            save_feedback(row)
            st.toast("Feedback Saved!")

    elif status == "Incorrect":
        st.write("Please provide the correct category:")
        c1, c2 = st.columns(2)
        actual_parent = c1.text_input("Correct Parent Category")
        actual_sub = c2.text_input("Correct Sub Category")
        
        if st.button("Save Feedback (Correction)"):
            # Log all predictions that were wrong
            top_pred = st.session_state['current_preds'][0] if st.session_state['current_preds'] else {'Parent': 'None', 'Sub': 'None'}
            row = {
                "product_name": st.session_state['p_name'],
                "product_description": st.session_state['p_desc'],
                "predicted_parent": top_pred['Parent'],
                "predicted_sub": top_pred['Sub'],
                "actual_parent": actual_parent,
                "actual_sub": actual_sub,
                "status": "Incorrect"
            }
            save_feedback(row)
            st.toast("Correction Saved!")

