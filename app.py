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

model = joblib.load("final_knn_model.pkl")
transformer = joblib.load("transformer.pkl")
xtrain_cols = joblib.load("xtrain_columns.pkl")



# ───────────────────────── Sidebar ─────────────────────────
st.sidebar.title("Navigation")
pages = [
    "Page 1: Introduction",
    "Page 2: Problem Statement & Objective",
    "Page 3: Data Collection",
    "Page 4: Exploratory Data Analysis (EDA)",
    "Page 5: Feature Engineering",
    "Page 6: Model Building",
    "Page 7: Fraud Detection UI"
]
selected_page = st.sidebar.radio("Go to", pages)

# ───────────────────────── Page 1 ─────────────────────────
if selected_page == "Page 1: Introduction":
    st.title("🔍 Credit Card Fraud Detection - Introduction")
    st.markdown("""
    - This project aims to detect fraudulent credit card transactions using machine learning.  
    - Fraud detection is a critical challenge in financial systems, as millions of transactions occur daily and only a tiny fraction are fraudulent.

    - In this project, we build a K-Nearest Neighbors (KNN) classification model to predict whether a transaction is fraudulent or legitimate.  
    - Unlike traditional models that use synthetic oversampling techniques like SMOTE, this project preserves the original data distribution to simulate a more realistic production scenario.  

    - The model is optimized using grid search and threshold tuning to improve fraud detection performance, focusing on key metrics such as precision, recall, and F1-score.  

    """)

# ───────────────────────── Page 2 ─────────────────────────
elif selected_page == "Page 2: Problem Statement & Objective":
    st.title("📌 Problem Statement & Objective")
    st.markdown("""
    ### 🧾 Problem Statement
    Credit card fraud continues to pose a serious threat to financial systems, costing billions of dollars each year. With the increasing volume of online transactions, detecting fraud in real-time is both essential and complex.
    One major challenge is the class imbalance—fraudulent transactions are extremely rare compared to legitimate ones, making them difficult to detect using standard machine learning approaches.

    This project addresses the challenge of identifying fraudulent credit card transactions by using a machine learning model trained on real-world transactional data, without relying on synthetic oversampling techniques like SMOTE. The goal is to build a solution that works effectively in real production settings where such techniques are often not used.


    ### 🎯 Objective
    The main objective of this project is to build an effective and interpretable K-Nearest Neighbors (KNN)-based classification model to:

    ✅ Classify credit card transactions as either fraudulent or legitimate

    ✅ Preserve real-world class imbalance to reflect true operational scenarios

    ✅ Use key features like transaction amount, merchant info, customer location, and job to make predictions

    ✅ Optimize the model using GridSearchCV and threshold tuning for improved precision, recall, and F1-score

    ✅ Deploy the final model through a Streamlit-based web application to enable real-time fraud detection based on user input
    """)

# ───────────────────────── Page 3 ─────────────────────────
elif selected_page == "Page 3: Data Collection":
    st.title("📊 Data Collection")
    
    st.markdown("""
    The dataset used in this project was sourced from an online open source platform.  
    It includes detailed records of individual transactions along with customer demographic information.

    ### 📋 All Columns in the Dataset:
    Below are all the original columns before cleaning:
    """)
    
    st.write(list(df.columns))

    st.markdown("""
    ### ✅ Key Features Description:
    | Feature | Description |
    |---------|-------------|
    | `Unnamed: 0` | Auto-generated index |
    | `trans_date_trans_time` | Transaction date and time |
    | `cc_num` | Credit card number (removed for privacy) |
    | `merchant` | Name of the merchant involved |
    | `category` | Transaction type (e.g., gas, shopping) |
    | `amt` | Transaction amount in ₹ |
    | `first`, `last` | Cardholder’s name (removed for privacy) |
    | `gender` | Cardholder’s gender |
    | `street`, `city`, `state`, `zip` | Location details |
    | `lat`, `long` | Geographic coordinates of cardholder |
    | `city_pop` | Population of the city |
    | `job` | Cardholder’s occupation |
    | `dob` | Date of birth (removed) |
    | `trans_num` | Transaction ID (removed for privacy) |
    | `unix_time` | Timestamp of transaction |
    | `merch_lat`, `merch_long` | Merchant’s geographic coordinates |
    | `is_fraud` | Target variable: 0 = Legitimate, 1 = Fraudulent |

    🛠️ **Additional Notes:**
    - Features like `cc_num`, `trans_num`, `dob`, `first`, and `last` were excluded to protect privacy.
    - The model is trained **without SMOTE**, keeping the **original class imbalance** to mimic real-world fraud detection conditions.
    - Feature engineering steps included encoding categorical variables, scaling numeric columns, and selecting relevant predictors for modeling.
    """)


# ───────────────────────── Page 4 ─────────────────────────
# ───────────────────────── Page 4 ─────────────────────────
elif selected_page == "Page 4: Exploratory Data Analysis (EDA)":
    import io
    buffer = io.StringIO()

    st.title("\U0001F4C8 Exploratory Data Analysis (EDA)")

    # 📋 Dataset Info
    st.markdown("### 📾 Data Info (Before Cleaning)")
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # 🔍 First Few Rows
    st.markdown("### 👀 First 5 Records")
    st.dataframe(df.head())

    # 🚨 Missing Values
    st.markdown("### ❓ Missing Values Summary")
    st.write(df.isnull().sum())
    
    
    # 🔐 Dropped Columns Explanation
    st.markdown("### 🔐 Dropped Columns: `cc_num`, `trans_num`, `dob`, `first`, `last`")
    st.info(
        """
        - `cc_num` (credit card number) is sensitive personally identifiable information (PII).
        - `trans_num` is a unique transaction ID with no predictive value.
        - `dob`, `first`, and `last` could lead to identity disclosure if included.
        
        👉 These columns were excluded to:
        - Protect user privacy and comply with ethical data handling.
        - Prevent overfitting or leakage due to unique identifiers.
        - Focus the model on features that generalize well to unseen data.
        """
    )
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    
    # ⚖️ Class Imbalance
    st.markdown("### ⚖️ Class Distribution")
    fraud_counts = df['is_fraud'].value_counts()
    st.write(fraud_counts)
    st.markdown(f"""
    - Legitimate Transactions: {fraud_counts[0]}
    - Fraudulent Transactions: {fraud_counts[1]}
    - Imbalance Ratio: 1 fraud per {round(fraud_counts[0] / fraud_counts[1], 2)} legit transactions
    """)

    fig, ax = plt.subplots()
    ax.pie(fraud_counts, labels=['Legit', 'Fraud'], autopct='%1.1f%%', startangle=90, colors=['skyblue', 'red'])
    ax.axis('equal')
    st.pyplot(fig)

    # 💰 Transaction Amount Distribution (< ₹500)
    st.markdown("### 💰 Transaction Amount Distribution (< ₹500)")
    fig, ax = plt.subplots()
    sns.histplot(df[df['amt'] < 500]['amt'], bins=50, kde=True, ax=ax, color="teal")
    ax.set_title("Transaction Amounts (< ₹500)")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # 📊 Fraud Count by Category
    st.markdown("### 📊 Fraud Count by Category")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x='category', hue='is_fraud', ax=ax, palette='rocket')
    ax.set_title("Fraud Distribution by Category")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

    # 📆 Boxplot of Transaction Amount by Fraud Status
    st.markdown("### 📆 Transaction Amount by Fraud Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='is_fraud', y='amt', data=df, ax=ax)
    ax.set_title("Transaction Amount by Fraud Class")
    ax.set_xlabel("Is Fraud")
    ax.set_ylabel("Amount")
    st.pyplot(fig)

    # 🔥 Correlation Heatmap
    st.markdown("### 🔥 Correlation Heatmap")
    num_cols = ['amt', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df[num_cols + ['is_fraud']].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

    # 📊 Pairplot (Sampled)
    st.markdown("### 📊 Pairwise Feature Plot (Sample)")
    sample_df = df.sample(5000, random_state=1)
    st.pyplot(sns.pairplot(sample_df, vars=['amt', 'city_pop', 'lat', 'long'], hue='is_fraud').fig)

    st.markdown("> These visuals help identify where frauds are concentrated and reveal patterns in numeric features.")


# ───────────────────────── Page 5 ─────────────────────────
elif selected_page == "Page 5: Feature Engineering":
    st.title("🔧 Feature Engineering")

    st.markdown("""
    ## 🔍 Overview

    Feature engineering transforms raw transaction data into meaningful inputs for the machine learning model.

    ---
    ### ✅ 1. Column Selection
    Selected the following features as input variables:
    
    - `amt`, `category`, `merchant`
    - `city`, `state`, `zip`
    - `job`, `gender`
    - `lat`, `long`, `city_pop`

    ---
    ### 🚫 2. Dropped Irrelevant Columns
    The following columns were removed due to privacy, redundancy, or irrelevance to prediction:
    
    - `cc_num`, `trans_num`, `dob`, `first`, `last`, `street`, `unix_time`, `merch_lat`, `merch_long`

    ---
    ### 🔤 3. Categorical Encoding
    All categorical variables were encoded using **Ordinal Encoding**:

    - `category`, `merchant`, `city`, `state`, `job`, `gender`

    ---
    ### 📏 4. Feature Scaling
    Scaling was applied using:
    
    - `RobustScaler` for: `amt`, `lat`, `long`, `city_pop` (to handle outliers)
    - `StandardScaler` for: `zip`

    ---
    ### ⚖️ 5. Handling Class Imbalance
    The original dataset is **highly imbalanced** (very few frauds).  
    However, **SMOTE was not used** in this version, to stay consistent with real-time fraud detection constraints.

    Instead, focus was placed on threshold tuning and classifier optimization.

    ---
    ### 🔄 6. Pipeline Construction
    All preprocessing (encoding + scaling) was applied using a `ColumnTransformer`.

    This was integrated with a **KNN Classifier** (with best parameters from `GridSearchCV`) into a reliable model pipeline.

    ✅ Final model saved as: `final_knn_model.pkl`  
    ✅ Preprocessing pipeline saved as: `scaler.pkl`
    """)


# ───────────────────────── Page 6 ─────────────────────────

elif selected_page == "Page 6: Model Building":
    st.title("🏗️ Model Building")

    st.markdown("""
    In this phase, we developed a robust machine learning pipeline using the **K-Nearest Neighbors (KNN)** algorithm to detect fraudulent credit card transactions.

    ---

    ### ✅ 1. Choice of Model
    We selected **K-Nearest Neighbors (KNN)** due to its:

    - Simplicity and interpretability  
    - No assumption about data distribution  
    - Capability to handle non-linear relationships  
    - Sensitivity to feature scaling (hence preprocessing is critical)

    ---

    ### ⚙️ 2. Preprocessing Steps
    The following transformations were applied:

    - **Categorical Encoding**: `OrdinalEncoder` for features like `category`, `merchant`, `city`, `state`, `job`, and `gender`
    - **Scaling**:
        - `RobustScaler` for: `amt`, `lat`, `long`, `city_pop` (robust to outliers)  
        - `StandardScaler` for: `zip` (to normalize ZIP code scale)

    These were implemented using a `ColumnTransformer`.

    ---

    ### 🔎 3. Hyperparameter Tuning
    We applied `GridSearchCV` to optimize KNN with 5-fold cross-validation:

    - `n_neighbors`: 3, 5, 7, 9  
    - `weights`: uniform, distance  
    - `p`: 1 (Manhattan), 2 (Euclidean)

    **Best Parameters Found**:
    - `n_neighbors`: 3  
    - `weights`: distance  
    - `p`: 1

    ---

    ### 📊 4. Model Evaluation
    The model was evaluated using the following metrics:

    - **Accuracy**
    - **Precision**
    - **Recall**
    - **F1 Score**

    ---

    ### 🧪 5. Final Results on Test Set

    - **Accuracy**: 0.996  
    - **Precision**: 0.70  
    - **Recall**: 0.57  
    - **F1 Score**: 0.63

    ✅ These metrics indicate a strong balance between detecting actual frauds and minimizing false positives.
    """)

# ───────────────────────── Page 7 ─────────────────────────
elif selected_page == "Page 7: Fraud Detection UI":
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
