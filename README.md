# 🌿 Plant Disease Classifier (Traditional ML)

A powerful web app for classifying plant leaf diseases using handcrafted features and traditional machine learning models like Random Forest, SVM, Gradient Boosting, and more.

---

## 🚀 Live Demo

*[🔗 Click here to try the app](https://plant-disease-app-2tgkxm6oc3d5r3wcuvxwgd.streamlit.app/) *

---

## 🧠 Features

* 🖼️ Upload plant leaf images (JPG, PNG)
* 🔍 Analyze and classify into 4 disease classes
* 🧬 Feature extraction using color & texture (LBP)
* 💡 Multiple ML models: RF, SVM, GBM, Voting, KNN, LR
* 📊 Model confidence visualizations
* 📜 Prediction history with download option
* 🎛️ UI controls for brightness/contrast + dark/light mode

---

## 🪴 Supported Disease Classes

* Healthy
* Multiple Diseases
* Rust
* Scab

---

## 🛠️ How It Works

1. Upload a plant leaf image
2. Image is enhanced (optional)
3. Features are extracted (RGB stats + LBP)
4. The selected model predicts disease class
5. Confidence and history are shown

---

## 📦 File Structure

```
plant-disease-app/
├── app.py                     # Streamlit app
├── requirements.txt          # Dependencies
├── README.md                 # Project readme
├── plant_disease_rf_model.joblib
├── plant_disease_svm_model.joblib
├── plant_disease_gb_model.joblib
├── plant_disease_voting_model.joblib
├── plant_disease_knn_model.joblib
└── plant_disease_logreg_model.joblib
```

---

## 🧪 Model Overview

| Model               | Type              | Notes                     |
| ------------------- | ----------------- | ------------------------- |
| Random Forest       | Bagging Ensemble  | Fast, robust              |
| SVM (RBF)           | Kernel Method     | Sharp decision boundaries |
| Gradient Boosting   | Boosting Ensemble | Most accurate             |
| Voting Classifier   | Soft Voting       | Combines top 3            |
| K-Nearest Neighbors | Lazy Learner      | For contrast & testing    |
| Logistic Regression | Linear Baseline   | For baseline comparison   |

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/plant-disease-app.git
cd plant-disease-app
```

### 2. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 📝 Notes

* For best results, upload clear images of single leaves
* App supports drag & drop or file picker upload
* Prediction report is downloadable as CSV

---

## 📄 License

MIT License *(add LICENSE file if missing)*

---

## 🙋‍♂️ Author

**Kratik Jain**
🌐 [LinkedIn]([https://linkedin.com/in/your-profile](https://www.linkedin.com/in/kratik-jain12/))
💌 Email: [kratikjain121@email.com](mailto:kratikjain121@email.com)

---

Made with ❤️ using Streamlit, OpenCV, and scikit-learn.
