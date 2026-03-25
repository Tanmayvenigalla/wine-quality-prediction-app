import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================================
# 🌈 PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide"
)

# ==========================================================
# 🎯 LOAD MODEL
# ==========================================================
@st.cache_resource
def load_model():
    try:
        if not os.path.exists("wine_model.pkl"):
            st.error("❌ Model file not found!")
            return None
        return joblib.load("wine_model.pkl")
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

# ==========================================================
# 📊 LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    try:
        if not os.path.exists("winequality-red.csv"):
            st.error("❌ Dataset file not found!")
            return pd.DataFrame()
        return pd.read_csv("winequality-red.csv")
    except Exception as e:
        st.error(f"❌ Data loading failed: {e}")
        return pd.DataFrame()

df = load_data()

# ==========================================================
# 🟨 SIDEBAR INPUTS
# ==========================================================
st.sidebar.title("⚙️ Wine Parameters")

fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.0)
volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5)
citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.3)
residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 2.0)
chlorides = st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05)
free_sulfur = st.sidebar.slider("Free Sulfur Dioxide", 1, 80, 15)
total_sulfur = st.sidebar.slider("Total Sulfur Dioxide", 6, 300, 50)
density = st.sidebar.slider("Density", 0.990, 1.005, 0.996)
pH = st.sidebar.slider("pH", 2.5, 4.5, 3.3)
sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.6)
alcohol = st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)

predict_clicked = st.sidebar.button("🍷 Predict Quality")

# ==========================================================
# 🧠 PREDICTION
# ==========================================================
if predict_clicked and model is not None:

    # Correct feature order
    feature_order = [
        "fixed acidity", "volatile acidity", "citric acid",
        "residual sugar", "chlorides", "free sulfur dioxide",
        "total sulfur dioxide", "density", "pH",
        "sulphates", "alcohol"
    ]

    input_df = pd.DataFrame([{
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur,
        "total sulfur dioxide": total_sulfur,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }])[feature_order]

    try:
        prediction = model.predict(input_df)[0]

        # Safe probability handling
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]
        else:
            prob = 0.5

        st.session_state.prediction = prediction
        st.session_state.prob = prob

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ==========================================================
# 🧭 NAVIGATION (Improved)
# ==========================================================
page = st.sidebar.radio(
    "📌 Navigate",
    ["Overview", "Dataset", "EDA", "Prediction"]
)

st.markdown("---")

# ==========================================================
# 🏠 OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("🍷 Wine Quality Prediction App")

    st.write("""
    This app predicts whether a wine is **Good (1)** or **Bad (0)**  
    using Machine Learning models.
    """)

# ==========================================================
# 📊 DATASET
# ==========================================================
elif page == "Dataset":

    if df.empty:
        st.warning("Dataset not available")
    else:
        st.subheader("Dataset Overview")
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write(df.describe())

# ==========================================================
# 🔍 EDA
# ==========================================================
elif page == "EDA":

    if df.empty:
        st.warning("Dataset not available")
    else:
        st.subheader("📊 Wine Quality Distribution")
        fig1 = px.histogram(df, x="quality", nbins=10)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("🔥 Correlation Heatmap")
        fig2 = px.imshow(df.corr(numeric_only=True), text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# 🍷 PREDICTION PAGE
# ==========================================================
elif page == "Prediction":

    st.subheader("🍷 Prediction Result")

    if "prediction" not in st.session_state:
        st.info("Enter values and click Predict")
    else:
        pred = st.session_state.prediction
        prob = st.session_state.prob

        if pred == 1:
            st.success(f"✅ Good Quality Wine (Confidence: {prob:.2f})")
        else:
            st.error(f"❌ Low Quality Wine (Confidence: {prob:.2f})")

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Quality Confidence (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        st.plotly_chart(fig)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("Wine Quality Prediction App 🍷 | ML + Streamlit")