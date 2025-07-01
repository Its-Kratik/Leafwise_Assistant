import streamlit as st
import numpy as np
import pandas as pd
import cv2
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from skimage.feature import local_binary_pattern

# Set page config
st.set_page_config(page_title="🌿 Plant Disease Classifier", page_icon="🌿", layout="centered")

# --- Theme Toggle ---
mode = st.sidebar.radio("🌗 Theme Mode", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
        .main-title {text-align: center; color: #8BC34A; font-size: 32px; font-weight: bold;}
        .subtitle {text-align: center; font-size: 18px; color: #cccccc; margin-bottom: 20px;}
        .stApp {background-color: #0f1117; color: white;}
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main-title {text-align: center; color: #4CAF50; font-size: 32px; font-weight: bold;}
        .subtitle {text-align: center; font-size: 18px; color: #333; margin-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)

st.markdown("<div class='main-title'>🌿 Plant Disease Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a plant leaf image to detect disease using classical ML models</div>", unsafe_allow_html=True)

# --- Load Models ---
models = {
    "Random Forest": joblib.load("plant_disease_rf_model.joblib"),
    "SVM (RBF Kernel)": joblib.load("plant_disease_svm_model.joblib"),
    "Gradient Boosting": joblib.load("plant_disease_gb_model.joblib"),
    "Voting Ensemble": joblib.load("plant_disease_voting_model.joblib"),
    "K-Nearest Neighbors": joblib.load("plant_disease_knn_model.joblib"),
    "Logistic Regression": joblib.load("plant_disease_logreg_model.joblib")
}

label_map = {0: 'Healthy', 1: 'Multiple Diseases', 2: 'Rust', 3: 'Scab'}

# --- Feature Extraction ---
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

# --- Sidebar Model Selection ---
st.sidebar.header("⚙️ Options")
selected_model_name = st.sidebar.selectbox("Choose Model", list(models.keys()))
selected_model = models[selected_model_name]

# --- Image Upload ---
uploaded_file = st.file_uploader("📁 Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.subheader("🖼️ Image Enhancements")
    brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
    contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

    enhancer_brightness = ImageEnhance.Brightness(image)
    image = enhancer_brightness.enhance(brightness)
    enhancer_contrast = ImageEnhance.Contrast(image)
    image = enhancer_contrast.enhance(contrast)

    st.image(image, caption="Enhanced Image", use_container_width=True)

    with st.spinner("🔍 Analyzing image..."):
        features = extract_features(image).reshape(1, -1)
        prediction = selected_model.predict(features)[0]
        probs = selected_model.predict_proba(features)[0]
        confidence = probs[prediction] * 100

    st.success(f"✅ Prediction: **{label_map[prediction]}**")
    st.info(f"📊 Confidence: {confidence:.2f}%  |  Model: {selected_model_name}")

    # Plot confidence
    st.subheader("🔬 Model Confidence")
    fig, ax = plt.subplots()
    ax.bar(label_map.values(), probs, color="#8BC34A")
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)

    # Session state to track history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Image": uploaded_file.name,
        "Prediction": label_map[prediction],
        "Confidence": f"{confidence:.2f}%",
        "Model": selected_model_name
    })

# --- Prediction History ---
if st.session_state.get("history"):
    st.subheader("📜 Prediction History")
    for item in reversed(st.session_state.history):
        st.markdown(f"- 🖼️ **{item['Image']}** → 🧠 {item['Prediction']} ({item['Confidence']}) via *{item['Model']}*")

    if st.button("⬇️ Export as CSV"):
        df = pd.DataFrame(st.session_state.history)
        st.download_button("Download CSV", df.to_csv(index=False), file_name="prediction_report.csv", mime="text/csv")

# --- Model Accuracy Chart ---
st.sidebar.subheader("📈 Model Benchmark")
model_scores = {
    "Random Forest": 0.93,
    "SVM (RBF Kernel)": 0.91,
    "Gradient Boosting": 0.94,
    "Voting Ensemble": 0.95,
    "K-Nearest Neighbors": 0.88,
    "Logistic Regression": 0.85
}

fig_score, ax_score = plt.subplots()
ax_score.bar(model_scores.keys(), model_scores.values(), color="#03DAC5")
ax_score.set_ylabel("Validation Accuracy")
ax_score.set_ylim([0.8, 1.0])
ax_score.set_xticklabels(model_scores.keys(), rotation=45, ha="right")
st.sidebar.pyplot(fig_score)

# --- Footer ---
st.markdown("""<hr style="border:1px solid gray">""", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Made with 💚 by Kratik Jain | Powered by Streamlit</div>", unsafe_allow_html=True)
