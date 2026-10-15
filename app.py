
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="FraudShield ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.joblib"

# --------------------------------------------------
# DARK FINTECH THEME
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #07111F;
    color: #EAF2FF;
}

[data-testid="stSidebar"] {
    background-color: #0B1728;
    border-right: 1px solid #244366;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    letter-spacing: 1px;
    text-shadow: 0 0 12px rgba(56, 139, 253, 0.45);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #C3D5ED;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3 {
    color: #F4F8FF !important;
}

p, label {
    color: #B8CBE4;
}

[data-testid="stMetric"] {
    background: #101F33;
    border: 1px solid #244366;
    padding: 18px;
    border-radius: 12px;
}

[data-testid="stMetricLabel"] {
    color: #B8CBE4 !important;
}

[data-testid="stMetricValue"] {
    color: #66B2FF !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: #1677FF;
    color: #FFFFFF !important;
    border: 1px solid #318BFF;
    border-radius: 9px;
    min-height: 44px;
    font-weight: 700;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #075BD8;
    border-color: #66B2FF;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #101F33;
    border-color: #294566;
}

hr {
    border-color: #244366;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EXISTING MODEL
# --------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Please check "
        "models/fraud_model.joblib."
    )
    st.stop()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# --------------------------------------------------
# SESSION-LEVEL PREDICTION HISTORY
# --------------------------------------------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# --------------------------------------------------
# SIDEBAR NAVIGATION
# No buttons modify the radio's session-state key.
# --------------------------------------------------
with st.sidebar:
    st.markdown("# 🛡️ FraudShield")
    st.caption("FINANCIAL SECURITY INTELLIGENCE")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Fraud Detection", "Model Monitoring"],
        key="main_navigation",
    )

    st.divider()
    st.markdown("**MODEL INFORMATION**")
    st.caption("Algorithm: XGBoost")
    st.caption("Experiment tracking: MLflow")
    st.caption("Frontend: Streamlit")

# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(100deg, #102947, #0B1728);
    border: 1px solid #244366;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 24px;
">
    <p style="
        color: #66B2FF;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 3px;
        margin-bottom: 6px;
    ">FINANCIAL SECURITY INTELLIGENCE</p>
    <h1 style="
        color: #FFFFFF !important;
        font-size: 34px;
        font-weight: 850;
        margin: 0;
    ">FraudShield ML</h1>
    <p style="color: #C3D5ED; margin-top: 8px; margin-bottom: 0;">
        Intelligent Financial Transaction Fraud Detection
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
if page == "Dashboard":
    st.subheader("Dashboard Overview")
    st.write(
        "Welcome to your financial transaction fraud detection platform."
    )

    history = st.session_state.prediction_history
    total = len(history)
    high_risk = sum(
        item["prediction"] == "High Risk" for item in history
    )
    lower_risk = total - high_risk

    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions This Session", total)
    col2.metric("High-Risk Predictions", high_risk)
    col3.metric("Lower-Risk Predictions", lower_risk)

    st.divider()
    st.subheader("Platform Modules")

    left, right = st.columns(2)

    with left:
        st.markdown("### 🔍 Fraud Detection")
        st.write(
            "Enter transaction details and estimate fraud probability "
            "using the trained XGBoost model."
        )
        st.caption(
            "Choose **Fraud Detection** from the sidebar to start."
        )

    with right:
        st.markdown("### 📊 Model Monitoring")
        st.write(
            "Review predictions recorded during the active session."
        )
        st.caption(
            "Choose **Model Monitoring** from the sidebar to review history."
        )

    st.divider()
    st.subheader("Technology Stack")

    tech1, tech2, tech3 = st.columns(3)
    tech1.info("**XGBoost**\n\nFraud classification model")
    tech2.info("**MLflow**\n\nExperiment tracking setup")
    tech3.info("**Streamlit**\n\nInteractive web interface")

# --------------------------------------------------
# FRAUD DETECTION
# --------------------------------------------------
elif page == "Fraud Detection":
    st.subheader("Transaction Risk Assessment")
    st.write(
        "Enter transaction details to estimate the transaction's fraud risk."
    )

    with st.form("transaction_form"):
        left, right = st.columns(2)

        with left:
            transaction_type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"],
            )

            amount = st.number_input(
                "Transaction Amount",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

            oldbalance_org = st.number_input(
                "Sender's Balance Before",
                min_value=0.0,
                value=5000.0,
                step=100.0,
            )

        with right:
            newbalance_orig = st.number_input(
                "Sender's Balance After",
                min_value=0.0,
                value=4000.0,
                step=100.0,
            )

            oldbalance_dest = st.number_input(
                "Receiver's Balance Before",
                min_value=0.0,
                value=0.0,
                step=100.0,
            )

            newbalance_dest = st.number_input(
                "Receiver's Balance After",
                min_value=0.0,
                value=1000.0,
                step=100.0,
            )

        submitted = st.form_submit_button(
            "🛡️ Check Transaction",
            use_container_width=True,
        )

    if submitted:
        transaction = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "newbalanceOrig": newbalance_orig,
            "oldbalanceDest": oldbalance_dest,
            "newbalanceDest": newbalance_dest,
        }])

        probability = float(
            model.predict_proba(transaction)[0][1]
        )

        prediction = (
            "High Risk" if probability >= 0.5 else "Lower Risk"
        )

        st.session_state.prediction_history.append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": transaction_type,
            "amount": amount,
            "probability": probability,
            "prediction": prediction,
        })

        st.divider()
        st.subheader("Prediction Results")

        result1, result2 = st.columns(2)
        result1.metric(
            "Estimated Fraud Probability",
            f"{probability * 100:.2f}%",
        )
        result2.metric("Risk Classification", prediction)

        st.progress(probability)

        if prediction == "High Risk":
            st.error(
                "⚠️ High Fraud Risk — Review this transaction."
            )
        else:
            st.success(
                "✅ Lower Fraud Risk — No fraud flagged by this model."
            )

        st.caption(
            "Educational prediction using synthetic data. "
            "The model cannot guarantee that a transaction is safe."
        )

# --------------------------------------------------
# MODEL MONITORING
# --------------------------------------------------
elif page == "Model Monitoring":
    st.subheader("Prediction Monitoring")
    st.write(
        "Review predictions recorded during the current session."
    )

    history = st.session_state.prediction_history
    high_risk = sum(
        item["prediction"] == "High Risk" for item in history
    )
    lower_risk = len(history) - high_risk

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", len(history))
    col2.metric("High Risk", high_risk)
    col3.metric("Lower Risk", lower_risk)

    st.divider()
    st.subheader("Recent Prediction History")

    if history:
        history_df = pd.DataFrame(history).copy()
        history_df["probability"] = history_df["probability"].map(
            lambda value: f"{value * 100:.2f}%"
        )
        history_df["amount"] = history_df["amount"].map(
            lambda value: f"{value:,.2f}"
        )

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No predictions recorded yet. Go to Fraud Detection "
            "and check a transaction."
        )

    st.divider()
    st.subheader("Monitoring Status")

    st.info(
        "Prediction history is currently maintained for the active "
        "session. Persistent prediction logging, live data-drift "
        "detection, and automatic MLflow metric synchronization "
        "are not yet implemented."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "FraudShield ML | XGBoost | MLflow | Streamlit | "
    "Educational project using synthetic transaction data"
)
