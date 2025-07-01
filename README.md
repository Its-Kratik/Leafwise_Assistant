# 🌿 Plant Disease Classifier (Traditional ML)

A web application for classifying plant leaf diseases using traditional machine learning techniques. Upload a plant leaf image, and the app predicts whether the leaf is healthy or affected by a disease (Multiple Diseases, Rust, or Scab).

## Features
- 🌱 **Easy-to-use web interface** powered by Streamlit
- 🖼️ **Image upload** for plant leaf analysis
- 🧠 **Traditional ML model** (Random Forest) for disease classification
- 📊 **Feature extraction** using color and texture (LBP) features

## How It Works
1. **Upload** a plant leaf image (JPG, JPEG, or PNG).
2. The app extracts features (mean/std RGB, LBP histogram) from the image.
3. A pre-trained Random Forest model predicts the disease class.
4. The result is displayed instantly.

## Disease Classes
- **Healthy**
- **Multiple Diseases**
- **Rust**
- **Scab**

## Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
cd plant-disease-app
```

### 2. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Open in Browser
Go to the local URL provided by Streamlit (usually http://localhost:8501).

## File Structure
- `app.py` — Main Streamlit app
- `plant_disease_rf_model.joblib` — Pre-trained Random Forest model
- `requirements.txt` — Python dependencies

## Model Details
- **Type:** Random Forest Classifier
- **Features:** Mean/Std RGB, Local Binary Pattern (LBP) histogram
- **Trained on:** Plant leaf images (details in your training pipeline)

## Notes
- The model file (`plant_disease_rf_model.joblib`) is required to run predictions.
- For best results, use clear images of single plant leaves.

## License
[MIT](LICENSE) (add a LICENSE file if needed)

---
*Made with ❤️ using Streamlit and scikit-learn.* 