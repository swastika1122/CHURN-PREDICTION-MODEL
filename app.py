import streamlit as st
import pandas as pd
import numpy as np
import joblib

# load model artifacts
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_order = joblib.load("feature_order.pkl")

def preprocess_churn(df):
    df = df.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    df.replace(
        ['null','NULL','na','NA','n/a','N/A','',' ','?','--'],
        np.nan,
        inplace=True
    )

    drop_cols = ['customer_id', 'churn_category', 'churn_reason']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

    for col, le in label_encoders.items():
        df[col] = le.transform(df[col].astype(str))

    df = df[feature_order]
    return df

# UI
st.title("📉 Customer Churn Prediction App")

uploaded_file = st.file_uploader("Upload Customer Data (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview", data.head())

    X = preprocess_churn(data)

    assert list(X.columns) == feature_order
    assert X.isna().sum().sum() == 0

    predictions = model.predict(X)
    data['Churn_Predicted'] = predictions

    st.success("Prediction Completed ✅")
    st.write(data)

    churned = data[data['Churn_Predicted'] == 1]
    st.write("🚨 Churned Customers", churned)
