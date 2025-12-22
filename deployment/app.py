# ===============================
# STREAMLIT UI FOR SAVINGS PREDICTION (FINAL FIX)
# ===============================

import streamlit as st

# 🚨 MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Personal Savings Prediction",
    layout="wide"
)

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ===============================
# PATHS
# ===============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_v1.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "random_forest_features_v1.pkl"

# ===============================
# LOAD MODEL & FEATURES
# ===============================
@st.cache_resource
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    return model, feature_names

model, FEATURE_NAMES = load_model_and_features()

# ===============================
# UI HEADER
# ===============================
st.title("💰 Personal Savings Prediction System")

st.markdown(
    """
This application predicts **Desired Savings** based on income and spending behavior  
using a **Random Forest Machine Learning model**.

It demonstrates **training–serving consistency**, a key MLOps principle.
"""
)

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.header("📥 Enter Financial Details")

Income = st.sidebar.number_input("Income (₹)", min_value=0, value=50000)
Age = st.sidebar.number_input("Age", min_value=18, value=30)
Dependents = st.sidebar.number_input("Dependents", min_value=0, value=1)

Rent = st.sidebar.number_input("Rent (₹)", min_value=0, value=12000)
Groceries = st.sidebar.number_input("Groceries (₹)", min_value=0, value=6000)
Transport = st.sidebar.number_input("Transport (₹)", min_value=0, value=3000)
Utilities = st.sidebar.number_input("Utilities (₹)", min_value=0, value=2500)
Healthcare = st.sidebar.number_input("Healthcare (₹)", min_value=0, value=2000)
Education = st.sidebar.number_input("Education (₹)", min_value=0, value=3000)
Entertainment = st.sidebar.number_input("Entertainment (₹)", min_value=0, value=2500)
Miscellaneous = st.sidebar.number_input("Miscellaneous (₹)", min_value=0, value=2000)

# ===============================
# RAW USER INPUT (PARTIAL FEATURES)
# ===============================
user_input = {
    "Income": Income,
    "Age": Age,
    "Dependents": Dependents,
    "Rent": Rent,
    "Groceries": Groceries,
    "Transport": Transport,
    "Utilities": Utilities,
    "Healthcare": Healthcare,
    "Education": Education,
    "Entertainment": Entertainment,
    "Miscellaneous": Miscellaneous
}

# ===============================
# ALIGN INPUT WITH TRAINING FEATURES
# ===============================
# Create empty dataframe with EXACT training features
model_input = pd.DataFrame(columns=FEATURE_NAMES)
model_input.loc[0] = 0.0  # default all missing features to 0

# Fill user-provided values where applicable
for feature, value in user_input.items():
    if feature in model_input.columns:
        model_input.at[0, feature] = value

# ===============================
# PREDICTION (✅ CORRECT WAY)
# ===============================
prediction = model.predict(model_input)[0]

# ===============================
# OUTPUT
# ===============================
st.subheader("📊 Prediction Result")
st.success(f"💡 **Predicted Desired Savings: ₹ {prediction:,.2f}**")

# ===============================
# VISUALIZATIONS
# ===============================
st.subheader("📈 Financial Insights")

expenses = {
    "Rent": Rent,
    "Groceries": Groceries,
    "Transport": Transport,
    "Utilities": Utilities,
    "Healthcare": Healthcare,
    "Education": Education,
    "Entertainment": Entertainment,
    "Miscellaneous": Miscellaneous
}

total_expenses = sum(expenses.values())

col1, col2 = st.columns(2)

# ---- PIE CHART ----
with col1:
    st.markdown("### 🧾 Expense Breakdown")
    fig1, ax1 = plt.subplots()
    ax1.pie(
        expenses.values(),
        labels=expenses.keys(),
        autopct="%1.1f%%",
        startangle=90
    )
    ax1.axis("equal")
    st.pyplot(fig1)

# ---- BAR CHART ----
with col2:
    st.markdown("### 💵 Income vs Expenses vs Savings")
    fig2, ax2 = plt.subplots()
    ax2.bar(
        ["Income", "Total Expenses", "Predicted Savings"],
        [Income, total_expenses, prediction]
    )
    ax2.set_ylabel("Amount (₹)")
    st.pyplot(fig2)

# ===============================
# FEATURE IMPORTANCE
# ===============================
st.subheader("🔍 Feature Importance (Model Explanation)")

importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURE_NAMES,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

fig3, ax3 = plt.subplots()
ax3.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
ax3.invert_yaxis()
ax3.set_xlabel("Importance Score")

st.pyplot(fig3)

st.markdown(
    """
**Explanation:**  
This chart shows how strongly each feature influenced the savings prediction.
"""
)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "📌 *This UI demonstrates deployment-safe inference using a trained ML model "
    "with strict feature schema alignment (MLOps best practice).*"
)
