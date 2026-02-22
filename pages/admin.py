import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Model Analytics", layout="wide")
st.title("📊 Model Performance Dashboard")

FEEDBACK_FILE = "feedback.csv"

# 1. Load Data
if not os.path.exists(FEEDBACK_FILE):
    st.warning("Waiting for data... (No feedback.csv found yet)")
    st.stop()

try:
    df = pd.read_csv(FEEDBACK_FILE)
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()

if df.empty:
    st.warning("Feedback file is empty.")
    st.stop()

# --- DATA PREPARATION ---
# Create 'Full Label' columns for easier charting (Parent > Sub)
# We use .get() to avoid errors if a column is temporarily missing
df['Actual_Full'] = df['actual_parent'].astype(str) + " > " + df['actual_sub'].astype(str)
df['Predicted_Full'] = df['predicted_parent'].astype(str) + " > " + df['predicted_sub'].astype(str)

# 2. Key Metrics
st.write(f"**Total Records:** {len(df)}")

correct_count = len(df[df['status'] == 'Correct'])
total_count = len(df)
accuracy = correct_count / total_count if total_count > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Live Accuracy", f"{accuracy:.1%}")
col2.metric("Total Corrections", len(df[df['status'] == 'Incorrect']))

# 3. Visualization: Correct vs Incorrect
fig_status = px.pie(df, names='status', title="Success Rate", 
                    color='status',
                    color_discrete_map={'Correct':'green', 'Incorrect':'red'})
col3.plotly_chart(fig_status, use_container_width=False)

st.divider()

# 4. Error Analysis
st.subheader("⚠️ Where is the model failing?")

errors = df[df['status'] == 'Incorrect']

if not errors.empty:
    # Chart: Which categories are hardest?
    # We count by the 'Actual_Full' label we created above
    hardest = errors['Actual_Full'].value_counts().reset_index()
    hardest.columns = ['Category', 'Error Count']
    
    st.bar_chart(hardest.set_index('Category'))
    
    # Table: Show details
    st.write("### Recent Mistakes")
    st.dataframe(
        errors[['product_name', 'Predicted_Full', 'Actual_Full']], 
        use_container_width=False,
        column_config={
            "product_name": "Product",
            "Predicted_Full": "AI Predicted",
            "Actual_Full": "Correct Label"
        }
    )
else:
    st.success("No errors reported yet! Great job.")

st.divider()

# 5. Raw Data Viewer
with st.expander("View Raw Data"):
    st.dataframe(df)