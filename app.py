import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medicine Treatment Cost Estimator",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .cost-card {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    .cost-amount {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load metadata and model
@st.cache_resource
def load_assets():
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
    meta_path = os.path.join(os.path.dirname(__file__), "model", "metadata.json")
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "metrics.json")
    
    pipeline = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return pipeline, metadata, metrics

try:
    pipeline, metadata, metrics = load_assets()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

cats = metadata["categorical_features"]
nums = metadata["numeric_features"]

# Header
st.markdown("<div class='main-title'>💊 Medicine Treatment Cost Estimator</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-title'>Predict the estimated <b>Payable Cost</b> of a medical treatment powered by a Random Forest model (R²: <b>{metrics['R2']:.3f}</b>, MAE: <b>₹{metrics['MAE']:.1f}</b>).</div>",
    unsafe_allow_html=True,
)

# Form Layout
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Patient Information")
    age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
    gender = st.selectbox("Gender", cats["Gender"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=24.5, step=0.1)
    chronic = st.selectbox("Chronic Condition?", cats["Chronic_Condition"])

with col2:
    st.subheader("🩺 Medicine Details")
    medicine_name = st.selectbox("Medicine Name", cats["Medicine_Name"])
    category = st.selectbox("Category", cats["Category"])
    med_type = st.selectbox("Form Factor / Type", cats["Type"])
    brand_type = st.selectbox("Branded or Generic", cats["Generic_or_Branded"])
    manufacturer = st.selectbox("Manufacturer", cats["Manufacturer"])
    dosage = st.number_input("Dosage (mg)", min_value=1, max_value=1000, value=20, step=5)
    quantity = st.number_input("Quantity (units)", min_value=1, max_value=200, value=30, step=1)

with col3:
    st.subheader("🏥 Treatment & Insurance")
    duration = st.number_input("Duration (Days)", min_value=1, max_value=180, value=15, step=1)
    severity = st.selectbox("Condition Severity", cats["Severity"])
    treatment_type = st.selectbox("Treatment Type", cats["Treatment_Type"])
    region = st.selectbox("Region", cats["Region"])
    insurance_pct = st.slider("Insurance Coverage (%)", min_value=0, max_value=100, value=30, step=5)
    discount_pct = st.slider("Pharmacy Discount (%)", min_value=0, max_value=100, value=10, step=5)

st.markdown("---")

# Feature engineering matching training pipeline
def prepare_input():
    row = {
        "Age": float(age),
        "Gender": gender,
        "BMI": float(bmi),
        "Chronic_Condition": chronic,
        "Medicine_Name": medicine_name,
        "Category": category,
        "Type": med_type,
        "Generic_or_Branded": brand_type,
        "Manufacturer": manufacturer,
        "Dosage_mg": float(dosage),
        "Quantity": float(quantity),
        "Duration_Days": float(duration),
        "Severity": severity,
        "Treatment_Type": treatment_type,
        "Region": region,
        "Insurance_Pct": float(insurance_pct),
        "Discount_Pct": float(discount_pct),
    }
    df = pd.DataFrame([row])

    age_bins = [0, 18, 35, 50, 65, 999]
    age_labels = ["Child", "Young Adult", "Adult", "Senior Adult", "Elderly"]
    df["Age_Group"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels, right=False)

    bmi_bins = [0, 18.5, 24.9, 29.9, 999]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["BMI_Category"] = pd.cut(df["BMI"], bins=bmi_bins, labels=bmi_labels, right=True)

    df["Has_Insurance"] = (df["Insurance_Pct"] > 0).astype(int)
    return df

res_col1, res_col2 = st.columns([1, 1.2])

with res_col1:
    calculate_btn = st.button("💰 Calculate Estimated Payable Cost", type="primary", use_container_width=True)

if calculate_btn:
    try:
        input_df = prepare_input()
        log_pred = pipeline.predict(input_df)[0]
        predicted_cost = float(np.expm1(log_pred))
        predicted_cost = max(predicted_cost, 0.0)

        with res_col2:
            st.markdown(
                f"""
                <div class="cost-card">
                    <div style="font-size: 1.1rem; opacity: 0.9;">Estimated Payable Cost</div>
                    <div class="cost-amount">₹ {predicted_cost:,.2f}</div>
                    <div style="font-size: 0.85rem; opacity: 0.85; margin-top: 0.5rem;">
                        {medicine_name} ({dosage}mg) • {quantity} units for {duration} days
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.info(
                f"ℹ️ **Breakdown**: Applied **{discount_pct}%** discount and **{insurance_pct}%** insurance coverage in **{region}** region."
            )
    except Exception as err:
        st.error(f"Prediction failed: {err}")

# Footer
st.markdown("---")
st.caption("ℹ️ Model trained on 5,000 treatment records. For estimation purposes only.")
