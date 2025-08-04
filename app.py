import os
from pathlib import Path
import zipfile
import pandas as pd
import streamlit as st

# ───────────────────────── Dataset Loader ─────────────────────────

import streamlit as st
import pandas as pd

# ✅ Cache the data loading
@st.cache_data
def load_data():
    df1 = pd.read_csv("fraudTrain.csv")
    df2 = pd.read_csv("fraudTest.csv")
    return pd.concat([df1, df2], ignore_index=True)

# ✅ Load combined dataset
df = load_data()


import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Note: Model and transformer are loaded within the button click handler
# model = joblib.load("final_knn_model.pkl")
# transformer = joblib.load("transformer.pkl")
# xtrain_cols = joblib.load("xtrain_columns.pkl")




# ───────────────────────── Page 7 ─────────────────────────
st.title("💳 Real-Time Fraud Detection")
st.subheader("🔧 Enter Transaction Details")

# ✅ User input fields (no indentation error here!)
user_input = {
    "amt": st.number_input("Transaction Amount (₹)", min_value=0.0, format="%.2f"),
    "category": st.selectbox("Category", sorted(df['category'].unique())),
    "city": st.text_input("City"),
    "state": st.text_input("State"),
    "job": st.text_input("Job"),
    "merchant": st.text_input("Merchant"),
    "gender": st.selectbox("Gender", ['M', 'F']),
    "zip": st.number_input("Zip Code", min_value=10000, max_value=99999),
    "city_pop": st.number_input("City Population", min_value=0),
    "lat": st.number_input("Latitude", format="%.6f"),
    "long": st.number_input("Longitude", format="%.6f"),
    "merch_lat": st.number_input("Merchant Latitude", format="%.6f"),
    "merch_long": st.number_input("Merchant Longitude", format="%.6f")
}

input_df = pd.DataFrame([user_input])

if st.button("🚨 Detect Fraud"):
    try:
        # ✅ Load model, transformer, and expected column names
        model = joblib.load("final_knn_model.pkl")
        transformer = joblib.load("transformer.pkl")
        xtrain_cols = joblib.load("xtrain_columns.pkl")

        # ✅ Transform and wrap the input
        transformed_input = transformer.transform(input_df)
        transformed_df = pd.DataFrame(transformed_input, columns=xtrain_cols)

        # ✅ Predict fraud probability
        y_proba = model.predict_proba(transformed_df)[:, 1]
        threshold = 0.99
        y_pred = (y_proba >= threshold).astype(int)

        # ✅ Show result
        st.write(f"**Fraud Probability:** {y_proba[0]:.4f}")
        if y_pred[0] == 1:
            st.error("⚠️ This transaction is likely **FRAUDULENT**.")
        else:
            st.success("✅ This transaction appears **legitimate**.")

    except FileNotFoundError as e:
        st.error(f"❌ Required file not found: `{e.filename}`")
    except Exception as e:
        st.error(f"❌ An error occurred during prediction:\n\n{str(e)}")
