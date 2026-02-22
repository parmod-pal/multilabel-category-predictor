import streamlit as st
import pandas as pd
import torch
import os
from transformers import AutoTokenizer, RobertaForSequenceClassification
from bs4 import BeautifulSoup
import re
import warnings
import time

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="AI Product Categorizer", page_icon="🛍️", layout="wide")
warnings.filterwarnings("ignore")

# --- 2. CONFIG ---
MODEL_PATH = "./saved_model"
FEEDBACK_FILE = "feedback.csv"
THRESHOLD = 0.5

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_resources():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

tokenizer, model, device = load_resources()

# --- 4. HELPERS ---
@st.cache_data
def get_all_categories():
    id2label = model.config.id2label
    parents = set()
    sub_map = {} 
    for label in id2label.values():
        parts = label.split(" > ")
        p = parts[0]
        s = parts[1] if len(parts) > 1 else "None"
        parents.add(p)
        if p not in sub_map: sub_map[p] = []
        if s not in sub_map[p]: sub_map[p].append(s)
    return sorted(list(parents)), sub_map

ALL_PARENTS, SUB_MAP = get_all_categories()

def clean_text(text):
    if not text: return ""
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator=" ")
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def save_feedback(data):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    df = pd.DataFrame([data])
    df.to_csv(FEEDBACK_FILE, mode='a', header=not file_exists, index=False)
    return True

# --- 5. THE MAGIC RESET FUNCTION ---
def reset_app():
    # This deletes ALL memory of what you typed and predicted
    st.session_state.clear()
    # This restarts the app (Making it look fresh)
    st.rerun()

# --- 6. PREDICTION LOGIC ---
def predict(name, desc):
    raw = (str(name) + " " + str(desc)).lower()
    clean = clean_text(name) + " " + clean_text(desc)
    
    inputs = tokenizer(clean, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
    id2label = model.config.id2label
    results = []
    
    sorted_idx = probs.argsort()[::-1]
    
    # Auto-Pick
    if probs[sorted_idx[0]] < THRESHOLD:
        idx = sorted_idx[0]
        parts = id2label[idx].split(" > ")
        results.append({"Parent": parts[0], "Sub": parts[1] if len(parts)>1 else "None", "Conf": probs[idx], "Note": "⚠️ Low Confidence"})
    else:
        for idx in sorted_idx:
            if probs[idx] >= THRESHOLD:
                parts = id2label[idx].split(" > ")
                results.append({"Parent": parts[0], "Sub": parts[1] if len(parts)>1 else "None", "Conf": probs[idx], "Note": "✅ Confident Match"})
            else: break
            
    # Rules
    existing = [f"{r['Parent']} > {r['Sub']}" for r in results]
    if any(x in raw for x in ["eco-friendly", "sustainable", "recycled"]):
        if "Sustainable > Sustainable" not in existing: results.append({"Parent": "Sustainable", "Sub": "Sustainable", "Conf": 1.0, "Note": "Rule"})
    if "tradeshow" in raw:
        if "Tradeshow > Tradeshow" not in existing: results.append({"Parent": "Tradeshow", "Sub": "Tradeshow", "Conf": 1.0, "Note": "Rule"})
    
    is_apparel = any(r['Parent'] == "Apparel" for r in results)
    if is_apparel:
        if any(x in raw for x in ["women's", "womens", "ladies"]): results.append({"Parent": "Gender", "Sub": "Women", "Conf": 1.0, "Note": "Rule"})
        elif any(x in raw for x in ["men's", "mens", "male"]): results.append({"Parent": "Gender", "Sub": "Men", "Conf": 1.0, "Note": "Rule"})
        elif "unisex" in raw: results.append({"Parent": "Gender", "Sub": "Unisex", "Conf": 1.0, "Note": "Rule"})
            
    return results

# --- 7. SIDEBAR ---
with st.sidebar:
    st.header("Admin Panel")
    try: st.page_link("pages/admin.py", label="Open Dashboard", icon="📊")
    except: pass
    st.divider()
    if os.path.exists(FEEDBACK_FILE):
        df = pd.read_csv(FEEDBACK_FILE)
        st.write(f"Total Records: **{len(df)}**")
        with open(FEEDBACK_FILE, "rb") as f:
            st.download_button("📥 Download CSV", f, "feedback.csv", "text/csv")

# --- 8. MAIN UI ---
st.title("🛍️ AI Product Categorizer")

# IMPORTANT: We use keys here. When reset_app() clears memory, these reset to empty.
if "p_name_input" not in st.session_state: st.session_state["p_name_input"] = ""
if "p_desc_input" not in st.session_state: st.session_state["p_desc_input"] = ""

with st.form("input_form"):
    p_name = st.text_input("Product Name", key="p_name_input")
    p_desc = st.text_area("Description", key="p_desc_input")
    submitted = st.form_submit_button("Predict Categories")

if submitted:
    if not p_name.strip() or not p_desc.strip():
        st.error("Please enter Name and Description.")
        st.stop()
    
    preds = predict(p_name, p_desc)
    st.session_state['current_preds'] = preds
    # We save these specifically for the feedback loop logic
    st.session_state['saved_name'] = p_name
    st.session_state['saved_desc'] = p_desc
    
    st.subheader("Predictions")
    for p in preds:
        color = "blue" if "Rule" in p['Note'] else "green" if "Confident" in p['Note'] else "orange"
        st.markdown(f"### :{color}[{p['Parent']} > {p['Sub']}]")
        st.caption(f"Confidence: {p['Conf']:.1%} | {p['Note']}")
        st.divider()

# --- 9. FEEDBACK LOOP ---
if 'current_preds' in st.session_state:
    st.write("### 📝 Rate this Prediction")
    status = st.radio("Is this correct?", ("Select", "Correct", "Incorrect"), horizontal=True)
    top = st.session_state['current_preds'][0]
    
    # 1. CORRECT BUTTON
    if status == "Correct":
        if st.button("Save Feedback (Correct)"):
            row = {
                "product_name": st.session_state['saved_name'],
                "product_description": st.session_state['saved_desc'],
                "predicted_parent": top['Parent'],
                "predicted_sub": top['Sub'],
                "actual_parent": top['Parent'],
                "actual_sub": top['Sub'],
                "status": "Correct"
                }
            save_feedback(row)
            st.success("✅ Saved!")
            time.sleep(2)
            # CALL THE RESET
            reset_app()

    # 2. INCORRECT BUTTON
    elif status == "Incorrect":
        c1, c2 = st.columns(2)
        sel_p = c1.selectbox("Parent", ["Select"] + ALL_PARENTS)
        subs = SUB_MAP.get(sel_p, []) if sel_p != "Select" else []
        sel_s = c2.selectbox("Sub", ["Select"] + subs)
        
        if st.button("Save Feedback (Correction)"):
            if sel_p == "Select": st.error("Select Parent Category")
            else:                
                row = {
                "product_name": st.session_state['saved_name'],
                "product_description": st.session_state['saved_desc'],
                "predicted_parent": top['Parent'],
                "predicted_sub": top['Sub'],
                "actual_parent": sel_p,
                "actual_sub": sel_s,
                "status": "Incorrect"
                }
                save_feedback(row)
                st.success("✅ Saved!")
                time.sleep(2)
                # CALL THE RESET
                reset_app()