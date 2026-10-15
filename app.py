
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FraudShield ML",
    page_icon="🛡️",
    layout="centered",
)

# Model file location
MODEL_PATH = Path("models/fraud_model.joblib")

# Application heading
st.title("🛡️ FraudShield ML")
st.subheader("Financial Transaction Fraud Detection")
st.write(
    "Enter transaction details to estimate the fraud risk."
)

# Check whether the trained model exists
if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please run python train.py first."
    )
    st.stop()


# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Transaction input fields
st.write("### Transaction Details")

transaction_type = st.selectbox(
    "Transaction Type",
    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0,
)

oldbalance_org = st.number_input(
    "Sender's Balance Before",
    min_value=0.0,
    value=5000.0,
)

newbalance_orig = st.number_input(
    "Sender's Balance After",
    min_value=0.0,
    value=4000.0,
)

oldbalance_dest = st.number_input(
    "Receiver's Balance Before",
    min_value=0.0,
    value=0.0,
)

newbalance_dest = st.number_input(
    "Receiver's Balance After",
    min_value=0.0,
    value=1000.0,
)

# Predict transaction risk
if st.button("Check Transaction", type="primary"):

    transaction = pd.DataFrame([
        {
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
        }
    ])

    probability = float(
        model.predict_proba(transaction)[0][1]
    )

    prediction = int(probability >= 0.5)

    # Display prediction results
    st.write("### Prediction Results")

    st.metric(
        "Estimated Fraud Probability",
        f"{probability * 100:.2f}%",
    )

    st.progress(probability)

    st.write(
        f"Raw model probability: {probability:.6f}"
    )

    st.write("Decision threshold: 50%")

    if prediction == 1:
        st.error(
            "⚠️ High Fraud Risk — Review this transaction."
        )
    else:
        st.success(
            "✅ Lower Fraud Risk — No fraud flagged by this model."
        )

    st.caption(
        "Educational prediction using synthetic data. "
        "This model cannot guarantee that a transaction is safe."
    )

# Footer
st.divider()
st.caption("FraudShield ML | XGBoost + MLflow + Streamlit")
