# =====================================
# CUSTOMER CHURN PREDICTION WEB APP
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# =====================================
# TITLE
# =====================================
st.title("📊 Customer Churn Prediction Dashboard")

# =====================================
# LOAD DATA (ULTRA FIXED)
# =====================================
def load_data():
    try:
        # Try normal read
        df = pd.read_csv("churn_data.csv")

        # If everything in one column → FIX IT
        if len(df.columns) == 1:
            df = pd.read_csv(
                "churn_data.csv",
                sep=",",
                engine="python",
                quoting=csv.QUOTE_NONE
            )

            # Still one column → manual split
            if len(df.columns) == 1:
                df = df[df.columns[0]].str.split(",", expand=True)
                df.columns = df.iloc[0]
                df = df[1:]

        # Clean column names
        df.columns = df.columns.str.strip()

        return df

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

df = load_data()

# =====================================
# FIX CHURN COLUMN
# =====================================
churn_col = None
for col in df.columns:
    if col.lower().strip() == "churn":
        churn_col = col
        break

if churn_col is None:
    st.error("❌ Churn column not found!")
    st.write("Detected columns:", df.columns)
    st.stop()

# Clean values
df[churn_col] = df[churn_col].astype(str).str.strip()

# Convert to numeric
df[churn_col] = df[churn_col].map({"Yes": 1, "No": 0})

# Rename
df.rename(columns={churn_col: "Churn"}, inplace=True)

# =====================================
# CLEANING
# =====================================
df.drop(columns=["customerID"], errors='ignore', inplace=True)

if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# =====================================
# ENCODING
# =====================================
df = pd.get_dummies(df, drop_first=True)

# =====================================
# SPLIT
# =====================================
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================
# MODEL
# =====================================
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📂 Dataset", "📊 Performance", "🤖 Prediction"])

# =====================================
# DATASET
# =====================================
if page == "📂 Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

# =====================================
# PERFORMANCE
# =====================================
elif page == "📊 Performance":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Accuracy")
        acc = accuracy_score(y_test, y_pred)
        st.success(f"{acc:.2f}")

    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

    st.subheader("Feature Importance")
    importances = rf.feature_importances_
    features = X.columns

    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    fig2, ax2 = plt.subplots()
    feat_imp.head(10).plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

# =====================================
# PREDICTION
# =====================================
elif page == "🤖 Prediction":
    st.subheader("Prediction Demo")

    if st.button("Predict Sample Customer"):
        sample = X_test.iloc[0].values
        pred = rf.predict([sample])[0]
        result = "Churn ❌" if pred == 1 else "No Churn ✅"
        st.success(f"Prediction: {result}")