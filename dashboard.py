import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_purchase_prediction_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "deployment_preprocessor.pkl"


# -----------------------------
# Load model and preprocessor
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)

    return model, preprocessor


model, preprocessor = load_model()


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Purchase Prediction Dashboard",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Intelligent Product Recommendation Dashboard")
st.caption("Customer Behaviour Based Purchase Prediction")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Customer Behaviour")

view_count = st.sidebar.number_input(
    "View Count",
    min_value=0,
    value=5,
)

cart_count = st.sidebar.number_input(
    "Cart Count",
    min_value=0,
    value=1,
)

remove_count = st.sidebar.number_input(
    "Remove Count",
    min_value=0,
    value=0,
)

price = st.sidebar.number_input(
    "Price",
    min_value=0.0,
    value=100.0,
)

brand_classes = list(preprocessor["brand_encoder"].classes_)
category_classes = list(preprocessor["category_encoder"].classes_)

brand = st.sidebar.selectbox(
    "Brand",
    brand_classes,
)

category = st.sidebar.selectbox(
    "Category",
    category_classes,
)


# -----------------------------
# Preprocessing function
# -----------------------------
def prepare_input():
    brand_encoded = preprocessor["brand_encoder"].transform([brand])[0]
    category_encoded = preprocessor["category_encoder"].transform([category])[0]

    data = pd.DataFrame(
        [
            {
                "view_count": view_count,
                "cart_count": cart_count,
                "remove_count": remove_count,
                "price": price,
                "brand": brand_encoded,
                "category_code": category_encoded,
            }
        ]
    )

    features = preprocessor["features"]

    data = data[features]

    scaled = preprocessor["scaler"].transform(data)

    return scaled, features


# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Purchase", type="primary"):

    X, features = prepare_input()

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])
    else:
        probability = float(prediction)

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            "Likely to Purchase" if prediction == 1 else "Unlikely to Purchase",
        )

    with col2:
        st.metric(
            "Purchase Probability",
            f"{probability:.2%}",
        )

    with col3:
        st.metric(
            "Price",
            f"₹{price:,.2f}",
        )


    # -----------------------------
    # Behaviour summary
    # -----------------------------
    st.subheader("Customer Behaviour")

    behaviour = pd.DataFrame(
        {
            "Behaviour": [
                "Views",
                "Cart Additions",
                "Removals",
            ],
            "Count": [
                view_count,
                cart_count,
                remove_count,
            ],
        }
    )

    st.bar_chart(
        behaviour.set_index("Behaviour")
    )


    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    st.subheader("SHAP Feature Explanation")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = np.asarray(shap_values)[0]

        explanation = pd.DataFrame(
            {
                "Feature": features,
                "SHAP Value": values,
            }
        ).sort_values(
            "SHAP Value",
            key=lambda x: abs(x),
            ascending=False,
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.barh(
            explanation["Feature"],
            explanation["SHAP Value"],
        )

        ax.set_xlabel("SHAP Value")
        ax.set_title("Feature Contribution to Prediction")

        plt.tight_layout()

        st.pyplot(fig)

    except Exception as e:
        st.info(
            f"SHAP explanation could not be generated for this model: {e}"
        )


# -----------------------------
# Metrics / Information
# -----------------------------
st.divider()

st.subheader("Model & System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Input Features", "6")

with col2:
    st.metric("Model", "Purchase Prediction Model")

with col3:
    st.metric("API", "FastAPI")


# -----------------------------
# Drift Check
# -----------------------------
st.subheader("Data Drift Check")

st.info(
    "The dashboard checks the current customer behaviour inputs against "
    "reasonable input ranges used by the application. Large or unusual "
    "values should be reviewed before making decisions."
)

drift_data = {
    "Feature": [
        "View Count",
        "Cart Count",
        "Remove Count",
        "Price",
    ],
    "Current Value": [
        view_count,
        cart_count,
        remove_count,
        price,
    ],
}

drift_df = pd.DataFrame(drift_data)

st.dataframe(
    drift_df,
    use_container_width=True,
)

st.caption(
    "Drift monitoring is intended as a screening indicator and does not "
    "replace formal statistical monitoring."
)