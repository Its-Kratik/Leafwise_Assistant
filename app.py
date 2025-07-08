import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="🌿 Plant Disease App", layout="wide")

# Load all models
models = {
    "Random Forest": joblib.load("plant_disease_rf_model.joblib"),
    "SVM (RBF Kernel)": joblib.load("plant_disease_svm_model.joblib"),
    "Gradient Boosting": joblib.load("plant_disease_gb_model.joblib"),
    "Voting Ensemble": joblib.load("plant_disease_voting_model.joblib"),
    "KNN": joblib.load("plant_disease_knn_model.joblib"),
    "Logistic Regression": joblib.load("plant_disease_logreg_model.joblib")
}

label_map = {0: 'Healthy', 1: 'Multiple Diseases', 2: 'Rust', 3: 'Scab'}

# Feature Extraction
@st.cache_data
def extract_features(pil_img):
    img = np.array(pil_img.resize((128, 128)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    return np.hstack([mean_rgb, std_rgb, lbp_hist])

# --- Page Sections ---
def show_about():
    st.title("🌿 About This Project")
    st.markdown("""
    This app classifies and detects plant leaf diseases using traditional ML models (no deep learning). 
    It uses handcrafted features like color stats and Local Binary Patterns (LBP) for classification.

    **Supported Diseases:**
    - Healthy
    - Multiple Diseases
    - Rust
    - Scab

    **Tech Used:** Streamlit, OpenCV, scikit-learn
    """)


def show_detection():
    st.title("🩺 Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"], key="detect")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
        features = extract_features(image).reshape(1, -1)
        model = models["Voting Ensemble"]
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[prediction] * 100
        st.success(f"Prediction: **{label_map[prediction]}**")
        st.info(f"Confidence: {confidence:.2f}%")


def show_classification():
    st.title("🧠 Plant Disease Classification")
    model_choice = st.selectbox("Choose Model", list(models.keys()))
    model = models[model_choice]
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"], key="classify")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)
        features = extract_features(image).reshape(1, -1)
        prediction = model.predict(features)[0]
        probs = model.predict_proba(features)[0]
        confidence = probs[prediction] * 100
        st.success(f"Predicted Class: **{label_map[prediction]}**")
        st.info(f"Confidence: {confidence:.2f}%")

        # Plot confidence
        st.subheader("Model Confidence")
        fig, ax = plt.subplots()
        ax.bar(label_map.values(), probs, color="#66BB6A")
        ax.set_ylabel("Probability")
        ax.set_ylim([0, 1])
        st.pyplot(fig)


def show_treatment():
    st.title("🌾 Treatment Guide")
    st.markdown("""
    ### 🍏 Apple and Pear Scab
    - **Symptoms**: Olive-green or black spots on leaves and fruits
    - **Treatment**: Use fungicides like Mancozeb; prune affected areas

    ### 🌿 Multiple Diseases
    - **General Advice**: Improve drainage and airflow, avoid overwatering

    ### 🍂 Rust
    - **Symptoms**: Yellow-orange spots on underside of leaves
    - **Treatment**: Use sulfur-based fungicides early in the season

    ### 🛡️ Healthy
    - Your plant appears healthy! Maintain regular care.
    """)


# --- Sidebar Navigation ---
st.sidebar.title("🔍 Navigation")
activity = st.sidebar.selectbox("Select Activity", ["About Project", "Plant Disease"])
if activity == "Plant Disease":
    task = st.sidebar.radio("Type", ["Detection", "Classification", "Treatment"])
else:
    task = None

# --- Render section ---
if activity == "About Project":
    show_about()
elif task == "Detection":
    show_detection()
elif task == "Classification":
    show_classification()
elif task == "Treatment":
    show_treatment()

# --- Footer ---
st.markdown("""<hr style="border:1px solid gray">""", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Made by Kratik Jain | Powered by Streamlit</div>", unsafe_allow_html=True)
