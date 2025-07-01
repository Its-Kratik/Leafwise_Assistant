import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import local_binary_pattern

# Load the trained model
model = joblib.load("plant_disease_rf_model.joblib")

# Label mapping
label_map = {0: 'Healthy', 1: 'Multiple Diseases', 2: 'Rust', 3: 'Scab'}

# Feature extractor
def extract_features_from_pil(pil_img):
    img = np.array(pil_img.resize((128, 128)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    return np.hstack([mean_rgb, std_rgb, lbp_hist])

# Streamlit UI
st.title("🌿 Plant Disease Classifier (Traditional ML)")

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract features & predict
    features = extract_features_from_pil(image).reshape(1, -1)
    prediction = model.predict(features)[0]
    st.success(f"🧠 Prediction: **{label_map[prediction]}**") 