import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Recruit Restaurant Visitor Forecasting", layout="wide")

def resolve_path(filename):
    """
    Finds the file in various possible locations for local and Hugging Face flexibility.
    """
    # Locations to check
    search_dirs = [
        os.path.join(os.getcwd(), 'models'),      # Local root execution
        os.path.join(os.getcwd(), 'src'),         # Root execution, files in src
        os.path.dirname(__file__),                # Same directory as script (HF typical)
        os.path.join(os.path.dirname(__file__), '../models'), # Local src execution
        os.getcwd()                               # Current working directory
    ]
    
    for directory in search_dirs:
        full_path = os.path.join(directory, filename)
        if os.path.exists(full_path):
            return full_path
    return None

@st.cache_resource
def load_resources():
    model_path = resolve_path('best_model.pkl')
    encoder_path = resolve_path('store_encoder.pkl')
    
    model = joblib.load(model_path) if model_path else None
    encoder = joblib.load(encoder_path) if encoder_path else None
            
    return model, encoder

model, encoder = load_resources()

st.title("🍽️ Recruit Restaurant Visitor Forecasting")
st.markdown("Predict future visitor numbers for restaurants.")

if model is None or encoder is None:
    st.error("Model or Encoder not found! Please check file locations.")
    # Debug info for user
    with st.expander("Path Debug Information"):
        st.write(f"CWD: {os.getcwd()}")
        st.write(f"Script Dir: {os.path.dirname(__file__)}")
        st.write(f"best_model.pkl found: {resolve_path('best_model.pkl')}")
        st.write(f"store_encoder.pkl found: {resolve_path('store_encoder.pkl')}")
else:
    tab1, tab2 = st.tabs(["🎯 Single Prediction", "📂 Batch Prediction"])
    
    # --- Single ---
    with tab1:
        st.header("Single Data Entry")
        
        col1, col2 = st.columns(2)
        with col1:
            # Dropdown for Store ID (using classes from encoder)
            store_ids = encoder.classes_
            selected_store = st.selectbox("Select Store ID", store_ids)
            input_date = st.date_input("Visit Date", datetime(2017, 4, 23))
            
        with col2:
            is_holiday = st.checkbox("Is Holiday?", value=False)
            
        if st.button("Predict Visitor Count"):
            # Prepare Input
            store_encoded = encoder.transform([selected_store])[0]
            year = input_date.year
            month = input_date.month
            day = input_date.day
            weekday = input_date.weekday()
            holiday_flg = 1 if is_holiday else 0
            
            features = np.array([[store_encoded, year, month, day, weekday, holiday_flg]])
            
            # Predict
            pred_log = model.predict(features)
            pred = np.expm1(pred_log)[0]
            pred = max(0, int(pred))
            
            st.success(f"Predicted Visitors: **{pred}**")
            st.metric("Visitors", pred)

    # --- Batch ---
    with tab2:
        st.header("Batch Prediction")
        st.write("Upload CSV with 'id' column (format: air_store_id_YYYY-MM-DD)")
        
        uploaded = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                if 'id' in df.columns:
                    # Parse ID
                    df['air_store_id'] = df['id'].apply(lambda x: '_'.join(x.split('_')[:2]))
                    df['visit_date'] = df['id'].apply(lambda x: x.split('_')[-1])
                    df['visit_date'] = pd.to_datetime(df['visit_date'])
                    
                    # Date Features
                    df['year'] = df['visit_date'].dt.year
                    df['month'] = df['visit_date'].dt.month
                    df['day'] = df['visit_date'].dt.day
                    df['day_of_week_num'] = df['visit_date'].dt.dayofweek
                    
                    # Holiday
                    df['holiday_flg'] = df['day_of_week_num'].apply(lambda x: 1 if x >= 5 else 0)
                    
                    # Encode
                    valid_stores = set(encoder.classes_)
                    df = df[df['air_store_id'].isin(valid_stores)].copy()
                    df['air_store_id_encoded'] = encoder.transform(df['air_store_id'])
                    
                    X_batch = df[['air_store_id_encoded', 'year', 'month', 'day', 'day_of_week_num', 'holiday_flg']]
                    
                    preds = np.expm1(model.predict(X_batch))
                    df['visitors'] = np.maximum(preds, 0).astype(int)
                    
                    st.subheader("Prediction Results")
                    st.dataframe(df[['id', 'visitors']].head(10))
                    
                    # Download
                    csv = df[['id', 'visitors']].to_csv(index=False).encode('utf-8')
                    st.download_button("Download Predictions", csv, "submission.csv", "text/csv")
                    
                else:
                    st.error("CSV must have 'id' column.")
            except Exception as e:
                st.error(f"Error: {e}")