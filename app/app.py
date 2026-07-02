import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Loan Repayment Prediction",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Loan Repayment Prediction System")
st.markdown(
    "Predict whether a customer is likely to **repay** the loan or **default**."
)

# -----------------------------
# Load Model & Scaler
# -----------------------------
model = load_model("../model/Model2.keras")
scaler = joblib.load("../model/scaler.pkl")
default_values = joblib.load("../model/default_values.pkl")
feature_order = joblib.load("../model/feature_order.pkl")

# -----------------------------
# Helper Function
# -----------------------------
def calculate_installment(loan_amount, annual_rate, months):

    monthly_rate = annual_rate / 100 / 12

    if monthly_rate == 0:
        return loan_amount / months

    installment = (
        loan_amount
        * monthly_rate
        * (1 + monthly_rate) ** months
    ) / ((1 + monthly_rate) ** months - 1)

    return installment


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("About")

st.sidebar.info(
    """
This project predicts whether a borrower is likely to repay a loan.

Model Used:
- Artificial Neural Network (ANN)

Dataset:
- Lending Club Loan Dataset

Only a few important details are required.
The remaining features are automatically filled using default values from the training dataset.
"""
)

st.sidebar.success("Ready for Prediction ✅")

# -----------------------------
# User Inputs
# -----------------------------
st.header("Enter Loan Details")

col1, col2 = st.columns(2)

with col1:

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500,
        max_value=50000,
        value=10000,
        step=500
    )

    int_rate = st.slider(
        "Interest Rate (%)",
        min_value=5.0,
        max_value=30.0,
        value=12.0,
        step=0.1
    )

    annual_inc = st.number_input(
        "Annual Income ($)",
        min_value=10000,
        value=60000,
        step=1000
    )

    emp_length = st.slider(
        "Employment Length (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

    dti = st.slider(
        "Debt-to-Income Ratio",
        min_value=0.0,
        max_value=40.0,
        value=18.0,
        step=0.1
    )

    # Added Subgrade Feature to dramatically increase prediction capabilities
    sub_grade = st.selectbox(
        "LendingClub Subgrade",
        [f"{letter}{num}" for letter in "ABCDEFG" for num in "12345"]
    )


with col2:

    credit_history_years = st.slider(
        "Credit History (Years)",
        min_value=0,
        max_value=40,
        value=10
    )

    total_acc = st.number_input(
        "Total Credit Accounts",
        min_value=1,
        value=20
    )

    open_acc = st.number_input(
        "Open Credit Accounts",
        min_value=1,
        value=10
    )

    revol_util = st.slider(
        "Revolving Utilization (%)",
        min_value=0.0,
        max_value=150.0,
        value=35.0,
        step=0.1
    )

    # Added dynamic input for Revolving Balance instead of fallback training static default values
    revol_bal = st.number_input(
        "Total Revolving Balance ($)",
        min_value=0,
        value=15000,
        step=500
    )

    term_option = st.selectbox(
        "Loan Term",
        ["36 months", "60 months"]
    )


# Automatically calculate installment
months = 36 if term_option == "36 months" else 60

installment = calculate_installment(
    loan_amnt,
    int_rate,
    months
)

# -----------------------------
# Advanced Options
# -----------------------------
with st.expander("Advanced Options"):

    mort_acc = st.number_input(
        "Mortgage Accounts",
        min_value=0,
        value=1
    )

    pub_rec = st.number_input(
        "Public Records",
        min_value=0,
        value=0
    )

    pub_rec_bankruptcies = st.number_input(
        "Public Record Bankruptcies",
        min_value=0,
        value=0
    )

    verification_status = st.selectbox(
        "Verification Status",
        [
            "Not Verified",
            "Verified",
            "Source Verified"
        ]
    )

    home_ownership = st.selectbox(
        "Home Ownership",
        [
            "Mortgage",
            "Own",
            "Rent",
            "Other"
        ]
    )

    purpose = st.selectbox(
        "Purpose of Loan",
        [
            "Debt Consolidation",
            "Credit Card",
            "Home Improvement",
            "Major Purchase",
            "Medical",
            "Moving",
            "Small Business",
            "Vacation",
            "House",
            "Educational",
            "Wedding",
            "Renewable Energy",
            "Other"
        ]
    )

# Predict button
predict_button = st.button(
    "Predict Loan Repayment",
    use_container_width=True
)

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    # Start with default values for all 79 features
    input_data = default_values.copy()

    # -----------------------------
    # Update Numerical Features
    # -----------------------------
    input_data["loan_amnt"] = loan_amnt
    input_data["term"] = months
    input_data["int_rate"] = int_rate
    input_data["installment"] = installment
    input_data["emp_length"] = emp_length
    input_data["annual_inc"] = annual_inc
    input_data["dti"] = dti
    input_data["open_acc"] = open_acc
    input_data["pub_rec"] = pub_rec
    input_data["revol_bal"] = revol_bal
    input_data["revol_util"] = revol_util
    input_data["total_acc"] = total_acc
    input_data["mort_acc"] = mort_acc
    input_data["pub_rec_bankruptcies"] = pub_rec_bankruptcies
    input_data["credit_history_years"] = credit_history_years

    # -----------------------------
    # Reset Dummy Variables
    # -----------------------------

    # Home Ownership
    input_data["OWN"] = 0
    input_data["RENT"] = 0
    input_data["OTHER"] = 0

    # Verification
    input_data["Verified"] = 0
    input_data["Source Verified"] = 0

    # Purpose
    purpose_columns = [
        "credit_card",
        "debt_consolidation",
        "educational",
        "home_improvement",
        "house",
        "major_purchase",
        "medical",
        "moving",
        "other",
        "renewable_energy",
        "small_business",
        "vacation",
        "wedding"
    ]

    for col in purpose_columns:
        input_data[col] = 0

    # Reset Subgrades One-Hot Variables
    # Reset Subgrade Columns
    for letter in "ABCDEFG":
        for num in "12345":
            sg_col = f"{letter}{num}"
            if sg_col in input_data:
                input_data[sg_col] = 0

    # -----------------------------
    # Home Ownership Encoding
    # -----------------------------
    if home_ownership == "Own":
        input_data["OWN"] = 1

    elif home_ownership == "Rent":
        input_data["RENT"] = 1

    elif home_ownership == "Other":
        input_data["OTHER"] = 1

    # Mortgage is the dropped category
    # so OWN=0, RENT=0, OTHER=0

    # -----------------------------
    # Verification Encoding
    # -----------------------------
    if verification_status == "Verified":
        input_data["Verified"] = 1

    elif verification_status == "Source Verified":
        input_data["Source Verified"] = 1

    # -----------------------------
    # Purpose Encoding
    # -----------------------------
    purpose_mapping = {
        "Credit Card": "credit_card",
        "Debt Consolidation": "debt_consolidation",
        "Educational": "educational",
        "Home Improvement": "home_improvement",
        "House": "house",
        "Major Purchase": "major_purchase",
        "Medical": "medical",
        "Moving": "moving",
        "Other": "other",
        "Renewable Energy": "renewable_energy",
        "Small Business": "small_business",
        "Vacation": "vacation",
        "Wedding": "wedding"
    }

    column = purpose_mapping.get(purpose)

    if column:
        input_data[column] = 1

    # -----------------------------
    # Subgrade Encoding
    # -----------------------------
    # A1 is the dropped category (all zeros)
    if sub_grade != "A1":
        input_data[sub_grade] = 1

    # -----------------------------
    # Create DataFrame
    # -----------------------------
    input_df = pd.DataFrame([input_data])

    # Ensure same column order as training
    input_df = input_df[feature_order]

    # -----------------------------
    # Scale Features
    # -----------------------------
    input_scaled = scaler.transform(input_df)

    # -----------------------------
    # Predict
    # -----------------------------
    probability = model.predict(input_scaled, verbose=0)[0][0]

    prediction = 1 if probability >= 0.5 else 0

    # -----------------------------
    # Display Results
    # -----------------------------
    st.markdown("---")
    st.header("Prediction Result")

    repay_probability = probability * 100
    default_probability = (1 - probability) * 100

    # -----------------------------
# Prediction Result
# -----------------------------

    if prediction == 1:

        if default_probability < 5:
            st.success("✅ The customer is highly likely to repay the loan.")

        elif default_probability < 15:
            st.success("✅ The customer is likely to repay the loan, although some risk is present.")

        elif default_probability < 25:
            st.warning("⚠️ The customer is predicted to repay the loan, but there is a moderate risk of default.")

        else:
            st.error("⚠️ The customer is predicted to repay the loan, but the risk of default is high. Additional review is recommended.")

    else:

        st.error("❌ The customer is likely to default on the loan.")

    st.subheader("Prediction Confidence")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            label="Repayment Probability",
            value=f"{repay_probability:.2f}%"
        )

    with col2:

        st.metric(
            label="Default Probability",
            value=f"{default_probability:.2f}%"
        )

    # -----------------------------
    # Risk Level
    # -----------------------------
    st.subheader("Risk Level")

    if default_probability < 5:

        st.success("🟢 Low Risk")

    elif default_probability < 15:

        st.warning("🟡 Medium Risk")

    elif default_probability < 25:

        st.warning("🟠 Medium to High Risk")

    else:

        st.error("🔴 High Risk")

    # -----------------------------
    # Loan Summary
    # -----------------------------
    st.subheader("Loan Summary")

    summary = pd.DataFrame({
        "Feature": [
            "Loan Amount",
            "Interest Rate",
            "LendingClub Subgrade",
            "Loan Term",
            "Monthly Installment",
            "Annual Income",
            "Employment Length",
            "Debt-to-Income Ratio",
            "Credit History",
            "Total Revolving Balance",
            "Open Accounts",
            "Total Accounts"
        ],
        "Value": [
            f"${loan_amnt:,.0f}",
            f"{int_rate:.2f}%",
            str(sub_grade),
            f"{months} months",
            f"${installment:.2f}",
            f"${annual_inc:,.0f}",
            f"{emp_length} years",
            f"{dti:.2f}",
            f"{credit_history_years} years",
            f"${revol_bal:,.0f}",
            open_acc,
            total_acc
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # -----------------------------
    # Recommendation
    # -----------------------------
    st.subheader("Recommendation")

    if prediction == 1:

        st.info(
            """
The applicant appears to have a **good probability of repaying** the loan.

**Recommendation:** Loan approval can be considered, subject to the lender's policies and any additional verification.
"""
        )

    else:

        st.warning(
            """
The applicant appears to have a **higher probability of defaulting** on the loan.

**Recommendation:** Consider additional verification, a reduced loan amount, or rejecting the application based on lending policies.
"""
        )

    # -----------------------------
    # Footer
    # -----------------------------
    st.markdown("---")
    st.caption(
        "Developed using Streamlit • TensorFlow ANN • Lending Club Loan Dataset"
    )