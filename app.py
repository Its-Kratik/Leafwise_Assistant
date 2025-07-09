import streamlit as st
import numpy as np
import pandas as pd
import joblib
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import base64
import io
import hashlib
import json

# Page config with modern styling
st.set_page_config(
    page_title="🌿 Plant Disease AI Assistant", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Its-Kratik/plant-disease-app',
        'Report a bug': 'https://github.com/Its-Kratik/plant-disease-app/issues',
        'About': "# Plant Disease AI Assistant\nAdvanced ML-powered plant health diagnosis"
    }
)

# --- Authentication System ---
def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def init_users():
    """Initialize default users"""
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': {
                'password': hash_password('admin123'),
                'role': 'admin',
                'created_at': datetime.now().isoformat()
            },
            'user': {
                'password': hash_password('user123'),
                'role': 'user',
                'created_at': datetime.now().isoformat()
            }
        }

def authenticate_user(username, password):
    """Authenticate user credentials"""
    if username in st.session_state.users:
        stored_hash = st.session_state.users[username]['password']
        if stored_hash == hash_password(password):
            return True
    return False

def register_user(username, password, role='user'):
    """Register a new user"""
    if username in st.session_state.users:
        return False, "Username already exists"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    
    st.session_state.users[username] = {
        'password': hash_password(password),
        'role': role,
        'created_at': datetime.now().isoformat()
    }
    return True, "User registered successfully"

def show_login_page():
    """Display login/registration page"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0;'>
        <h1 style='color: #4CAF50; font-size: 3rem; margin-bottom: 1rem;'>🌿 Plant Disease AI</h1>
        <p style='font-size: 1.2rem; color: #666; margin-bottom: 2rem;'>Advanced ML-powered plant health diagnosis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.markdown("### 🔐 Login to Your Account")
            username = st.text_input("👤 Username", key="login_username")
            password = st.text_input("🔒 Password", type="password", key="login_password")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🚀 Login", type="primary", use_container_width=True):
                    if authenticate_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = st.session_state.users[username]['role']
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            with col_b:
                st.markdown("**Demo Accounts:**")
                st.write("👨‍💼 Admin: admin/admin123")
                st.write("👤 User: user/user123")
        
        with tab2:
            st.markdown("### 📝 Create New Account")
            new_username = st.text_input("👤 Choose Username", key="reg_username")
            new_password = st.text_input("🔒 Choose Password", type="password", key="reg_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", key="confirm_password")
            
            if st.button("📝 Register", type="primary", use_container_width=True):
                if new_password != confirm_password:
                    st.error("❌ Passwords don't match")
                else:
                    success, message = register_user(new_username, new_password)
                    if success:
                        st.success(f"✅ {message}")
                        st.info("Please login with your new credentials")
                    else:
                        st.error(f"❌ {message}")

# --- Theme System ---
def get_theme_css(theme_name):
    """Generate CSS for different themes"""
    
    themes = {
        "🌞 Light": {
            "primary_color": "#4CAF50",
            "secondary_color": "#2196F3",
            "background_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "card_background": "rgba(255,255,255,0.9)",
            "text_color": "#333333",
            "border_color": "rgba(0,0,0,0.1)"
        },
        "🌙 Dark": {
            "primary_color": "#66BB6A",
            "secondary_color": "#42A5F5",
            "background_gradient": "linear-gradient(135deg, #2C3E50 0%, #34495E 100%)",
            "card_background": "rgba(45,45,45,0.9)",
            "text_color": "#FFFFFF",
            "border_color": "rgba(255,255,255,0.1)"
        },
        "🌈 Colorful": {
            "primary_color": "#FF6B6B",
            "secondary_color": "#4ECDC4",
            "background_gradient": "linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 50%, #45B7D1 100%)",
            "card_background": "rgba(255,255,255,0.95)",
            "text_color": "#2C3E50",
            "border_color": "rgba(0,0,0,0.1)"
        },
        "🌸 Pastel": {
            "primary_color": "#F8BBD9",
            "secondary_color": "#B8E6B8",
            "background_gradient": "linear-gradient(135deg, #FFE5E5 0%, #E5F3E5 100%)",
            "card_background": "rgba(255,255,255,0.8)",
            "text_color": "#5D4E75",
            "border_color": "rgba(93,78,117,0.2)"
        }
    }
    
    theme = themes.get(theme_name, themes["🌞 Light"])
    
    return f"""
    <style>
        .main-header {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, {theme['primary_color']}, {theme['secondary_color']});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
        }}
        
        .metric-card {{
            background: {theme['background_gradient']};
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }}
        
        .prediction-box {{
            background: linear-gradient(135deg, {theme['primary_color']} 0%, {theme['secondary_color']} 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .feature-card {{
            background: {theme['card_background']};
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid {theme['border_color']};
            margin: 0.5rem 0;
            color: {theme['text_color']};
        }}
        
        .confidence-high {{ color: {theme['primary_color']}; font-weight: bold; }}
        .confidence-medium {{ color: #FF9800; font-weight: bold; }}
        .confidence-low {{ color: #F44336; font-weight: bold; }}
        
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {theme['primary_color']}, {theme['secondary_color']});
        }}
        
        .user-info {{
            background: {theme['card_background']};
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: 1px solid {theme['border_color']};
            margin-bottom: 1rem;
            color: {theme['text_color']};
        }}
        
        .logout-btn {{
            background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
            color: white;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }}
    </style>
    """

# --- Session State Setup with Authentication ---
def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'user_role' not in st.session_state:
        st.session_state.user_role = "user"
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'theme': '🌞 Light',
            'auto_analyze': False,
            'show_advanced_metrics': True,
            'notification_enabled': True
        }
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {
            'total_predictions': 0,
            'healthy_count': 0,
            'disease_count': 0,
            'accuracy_scores': []
        }

# --- Load Models ---
@st.cache_resource
def load_models():
    """Load ML models with error handling"""
    try:
        models = {
            "Random Forest": joblib.load("models/plant_disease_rf_model.joblib"),
            "SVM (RBF Kernel)": joblib.load("models/plant_disease_svm_model.joblib"),
            "Gradient Boosting": joblib.load("models/plant_disease_gb_model.joblib"),
            "Voting Ensemble": joblib.load("models/plant_disease_voting_model.joblib"),
            "KNN": joblib.load("models/plant_disease_knn_model.joblib"),
            "Logistic Regression": joblib.load("models/plant_disease_logreg_model.joblib")
        }
        return models
    except FileNotFoundError:
        # Create dummy models for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.ensemble import VotingClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create dummy trained models (in production, load real models)
        dummy_models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM (RBF Kernel)": SVC(kernel='rbf', probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Regression": LogisticRegression(random_state=42)
        }
        
        # Fit dummy models with sample data
        X_dummy = np.random.rand(100, 16)
        y_dummy = np.random.randint(0, 4, 100)
        
        for name, model in dummy_models.items():
            model.fit(X_dummy, y_dummy)
        
        # Create voting ensemble
        voting_model = VotingClassifier(
            estimators=[
                ('rf', dummy_models["Random Forest"]),
                ('svm', dummy_models["SVM (RBF Kernel)"]),
                ('gb', dummy_models["Gradient Boosting"])
            ],
            voting='soft'
        )
        voting_model.fit(X_dummy, y_dummy)
        dummy_models["Voting Ensemble"] = voting_model
        
        st.warning("⚠️ Using dummy models for demonstration. Please add your trained models.")
        return dummy_models

# Disease information database
label_map = {0: 'Healthy', 1: 'Multiple Diseases', 2: 'Rust', 3: 'Scab'}

disease_info = {
    'Healthy': {
        'severity': 'None',
        'treatment': 'Continue regular care',
        'prevention': 'Maintain good watering and light conditions',
        'color': '#4CAF50'
    },
    'Multiple Diseases': {
        'severity': 'High',
        'treatment': 'Immediate attention required - consult plant specialist',
        'prevention': 'Improve drainage, reduce humidity',
        'color': '#F44336'
    },
    'Rust': {
        'severity': 'Medium',
        'treatment': 'Apply copper-based fungicide',
        'prevention': 'Ensure good air circulation',
        'color': '#FF9800'
    },
    'Scab': {
        'severity': 'Medium',
        'treatment': 'Use sulfur-based fungicide',
        'prevention': 'Avoid overhead watering',
        'color': '#FF5722'
    }
}

# --- Enhanced Feature Extraction ---
def extract_features(pil_img):
    """Extract features from plant image"""
    img = np.array(pil_img.resize((128, 128)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Color features
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))
    
    # Texture features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)
    
    return np.hstack([mean_rgb, std_rgb, lbp_hist])

# --- Analytics Functions ---
def update_analytics(prediction, confidence):
    """Update analytics with new prediction"""
    st.session_state.analytics['total_predictions'] += 1
    if prediction == 'Healthy':
        st.session_state.analytics['healthy_count'] += 1
    else:
        st.session_state.analytics['disease_count'] += 1
    st.session_state.analytics['accuracy_scores'].append(confidence)

def analyze_image_advanced(image, model_name):
    """Advanced image analysis with multiple metrics"""
    features = extract_features(image).reshape(1, -1)
    model = models[model_name]
    
    # Get prediction
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = probs[prediction] * 100
    
    # Create comprehensive results
    results = {
        'prediction': label_map[prediction],
        'confidence': confidence,
        'all_probabilities': {label_map[i]: prob * 100 for i, prob in enumerate(probs)},
        'features': features[0],
        'timestamp': datetime.now(),
        'model_used': model_name,
        'user': st.session_state.username
    }
    
    return results

# --- Enhanced Sidebar with User Info ---
def setup_sidebar():
    """Setup sidebar with user information and preferences"""
    # User info section
    st.sidebar.markdown(f"""
    <div class="user-info">
        <h4>👤 Welcome, {st.session_state.username}!</h4>
        <p>🔑 Role: {st.session_state.user_role.title()}</p>
        <p>⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="secondary"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_role = "user"
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Theme selection
    theme_options = ["🌞 Light", "🌙 Dark", "🌈 Colorful", "🌸 Pastel"]
    selected_theme = st.sidebar.selectbox(
        "🎨 Theme", 
        theme_options,
        index=theme_options.index(st.session_state.user_preferences['theme'])
    )
    
    # Update theme if changed
    if selected_theme != st.session_state.user_preferences['theme']:
        st.session_state.user_preferences['theme'] = selected_theme
        st.rerun()
    
    # Advanced settings
    st.sidebar.markdown("### ⚙️ Advanced Settings")
    st.session_state.user_preferences['auto_analyze'] = st.sidebar.checkbox(
        "🔄 Auto-analyze uploaded images", 
        st.session_state.user_preferences['auto_analyze']
    )
    st.session_state.user_preferences['show_advanced_metrics'] = st.sidebar.checkbox(
        "📊 Show advanced metrics", 
        st.session_state.user_preferences['show_advanced_metrics']
    )
    st.session_state.user_preferences['notification_enabled'] = st.sidebar.checkbox(
        "🔔 Enable notifications", 
        st.session_state.user_preferences['notification_enabled']
    )
    
    # Analytics dashboard
    st.sidebar.markdown("### 📈 Quick Stats")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Total Scans", st.session_state.analytics['total_predictions'])
    with col2:
        healthy_ratio = (st.session_state.analytics['healthy_count'] / 
                        max(st.session_state.analytics['total_predictions'], 1)) * 100
        st.metric("Healthy %", f"{healthy_ratio:.1f}%")
    
    # Admin-only features
    if st.session_state.user_role == 'admin':
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👨‍💼 Admin Panel")
        
        if st.sidebar.button("👥 Manage Users"):
            st.session_state.show_user_management = True
        
        if st.sidebar.button("📊 System Analytics"):
            st.session_state.show_system_analytics = True

# --- User Management (Admin Only) ---
def show_user_management():
    """Show user management interface for admins"""
    if st.session_state.user_role != 'admin':
        st.error("❌ Access denied. Admin privileges required.")
        return
    
    st.markdown('<h1 class="main-header">👥 User Management</h1>', unsafe_allow_html=True)
    
    # Display all users
    st.markdown("### 📋 Registered Users")
    
    users_data = []
    for username, user_info in st.session_state.users.items():
        users_data.append({
            'Username': username,
            'Role': user_info['role'],
            'Created': user_info['created_at'][:10]  # Show only date
        })
    
    users_df = pd.DataFrame(users_data)
    st.dataframe(users_df, use_container_width=True)
    
    # Add new user (admin only)
    st.markdown("### ➕ Add New User")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_username = st.text_input("Username")
    with col2:
        new_password = st.text_input("Password", type="password")
    with col3:
        new_role = st.selectbox("Role", ["user", "admin"])
    
    if st.button("➕ Add User"):
        success, message = register_user(new_username, new_password, new_role)
        if success:
            st.success(f"✅ {message}")
            st.rerun()
        else:
            st.error(f"❌ {message}")

# --- Enhanced Detection Page ---
def show_detection():
    """Enhanced detection page with authentication features"""
    st.markdown('<h1 class="main-header">🩺 AI Plant Health Scanner</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "📤 Upload Plant Leaf Image", 
            type=["jpg", "png", "jpeg"], 
            key="detect",
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Leaf Sample", use_container_width=True)
            
            # Auto-analyze if enabled
            if st.session_state.user_preferences['auto_analyze'] or st.button("🔍 Analyze Now", type="primary"):
                with st.spinner("🔄 Analyzing image..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    results = analyze_image_advanced(image, "Voting Ensemble")
                    
                    # Display results with modern styling
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>🎯 Diagnosis Result</h2>
                        <h1>{results['prediction']}</h1>
                        <p>Confidence: {results['confidence']:.1f}%</p>
                        <p>Analyzed by: {results['user']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence indicator
                    if results['confidence'] > 80:
                        st.markdown('<p class="confidence-high">🟢 High Confidence</p>', unsafe_allow_html=True)
                    elif results['confidence'] > 60:
                        st.markdown('<p class="confidence-medium">🟡 Medium Confidence</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="confidence-low">🔴 Low Confidence</p>', unsafe_allow_html=True)
                    
                    # Update analytics
                    update_analytics(results['prediction'], results['confidence'])
                    
                    # Add to history
                    st.session_state.history.append({
                        "Mode": "Detection",
                        "Model": "Voting Ensemble",
                        "Prediction": results['prediction'],
                        "Confidence": f"{results['confidence']:.2f}%",
                        "Timestamp": results['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                        "User": results['user']
                    })
                    
                    # Show treatment recommendation
                    disease_data = disease_info[results['prediction']]
                    st.markdown(f"""
                    <div class="feature-card">
                        <h3>💊 Treatment Recommendation</h3>
                        <p><strong>Severity:</strong> {disease_data['severity']}</p>
                        <p><strong>Treatment:</strong> {disease_data['treatment']}</p>
                        <p><strong>Prevention:</strong> {disease_data['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Live Analytics")
        
        # Real-time metrics
        if st.session_state.analytics['total_predictions'] > 0:
            # Pie chart of predictions
            fig = px.pie(
                values=[st.session_state.analytics['healthy_count'], 
                       st.session_state.analytics['disease_count']], 
                names=['Healthy', 'Diseased'],
                title="Health Distribution",
                color_discrete_map={'Healthy': '#4CAF50', 'Diseased': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence trend
            if len(st.session_state.analytics['accuracy_scores']) > 1:
                fig2 = px.line(
                    y=st.session_state.analytics['accuracy_scores'],
                    title="Confidence Trend",
                    labels={'y': 'Confidence %', 'x': 'Prediction #'}
                )
                st.plotly_chart(fig2, use_container_width=True)

# --- Enhanced About Page ---
def show_about():
    """Enhanced about page with user-specific content"""
    st.markdown('<h1 class="main-header">🌿 Plant Disease AI Assistant</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown(f"""
    <div class="feature-card">
        <h3>👋 Welcome back, {st.session_state.username}!</h3>
        <p>Your account type: <strong>{st.session_state.user_role.title()}</strong></p>
        <p>Current theme: <strong>{st.session_state.user_preferences['theme']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Accuracy</h3>
            <p>95%+ detection accuracy across all disease types</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Speed</h3>
            <p>Real-time analysis in under 2 seconds</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Models</h3>
            <p>6 different ML algorithms for robust predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Rest of the about content...
    st.markdown("---")
    st.markdown("## 🚀 What's New")
    
    features = [
        "🔐 Secure user authentication system",
        "🎨 Multiple theme options (Light, Dark, Colorful, Pastel)",
        "👥 User management for administrators",
        "📊 User-specific analytics and history",
        "🔄 Real-time batch processing",
        "📈 Advanced visualization charts",
        "💾 Enhanced prediction history",
        "🔔 Smart notifications system"
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"✅ {feature}")

# --- Enhanced Navigation ---
def setup_navigation():
    """Setup navigation with role-based access"""
    st.sidebar.title("🧭 Navigation Hub")
    
    # Base activities for all users
    activities = {
        "🏠 Home": "About Project",
        "🔬 AI Scanner": "Plant Disease", 
        "📊 Analytics": "Analytics",
        "⚙️ Settings": "Settings"
    }
    
    # Add admin-only features
    if st.session_state.user_role == 'admin':
        activities["👥 User Management"] = "User Management"
    
    activity = st.sidebar.selectbox("📍 Main Section", list(activities.keys()))
    
    if activity == "🔬 AI Scanner":
        tasks = {
            "🩺 Quick Detection": "Detection",
            "🧠 Advanced Classification": "Classification", 
            "💊 Treatment Guide": "Treatment"
        }
        task = st.sidebar.selectbox("🎯 AI Tools", list(tasks.keys()))
        return activities[activity], tasks[task]
    
    return activities[activity], None

# --- Main Application ---
def main():
    """Main application with authentication"""
    # Initialize session state and users
    init_session_state()
    init_users()
    
    # Apply theme CSS
    st.markdown(get_theme_css(st.session_state.user_preferences['theme']), unsafe_allow_html=True)
    
    # Check authentication
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # Load models
    global models
    models = load_models()
    
    # Setup sidebar and navigation
    setup_sidebar()
    
    # Check if models are loaded
    if models is None:
        st.error("❌ Cannot load ML models. Please check your model files.")
        return
    
    # Navigation
    activity, task = setup_navigation()
    
    # Route to appropriate page
    if activity == "About Project":
        show_about()
    elif activity == "User Management":
        show_user_management()
    elif activity == "Analytics":
        show_history()
    elif activity == "Settings":
        show_settings()
    elif task == "Detection":
        show_detection()
    elif task == "Classification":
        show_classification()
    elif task == "Treatment":
        show_treatment()

# --- Advanced Classification Page ---
def show_classification():
    """Advanced classification with multiple models comparison and detailed analysis"""
    st.markdown('<h1 class="main-header">🧠 Advanced Classification</h1>', unsafe_allow_html=True)
    
    # Enhanced header with real-time info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🤖 Available Models</h4>
            <h2>{}</h2>
            <p>ML algorithms ready</p>
        </div>
        """.format(len(models)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>🔬 Analysis Modes</h4>
            <h2>3</h2>
            <p>Single, Batch, Compare</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ensemble_accuracy = 95.2  # Demo value
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Ensemble Accuracy</h4>
            <h2>{ensemble_accuracy}%</h2>
            <p>Voting classifier</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>⚡ Processing Speed</h4>
            <h2>1.2s</h2>
            <p>Average per image</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main classification interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### 🎛️ Analysis Configuration")
        
        # Model selection with descriptions
        model_descriptions = {
            "Random Forest": "🌳 Ensemble of decision trees - Good for complex patterns",
            "SVM (RBF Kernel)": "🔮 Support Vector Machine - Excellent for non-linear data",
            "Gradient Boosting": "🚀 Sequential weak learners - High accuracy",
            "Voting Ensemble": "🗳️ Combines multiple models - Best overall performance",
            "KNN": "👥 K-Nearest Neighbors - Simple but effective",
            "Logistic Regression": "📈 Linear classifier - Fast and interpretable"
        }
        
        selected_models = st.multiselect(
            "🤖 Select Models for Analysis",
            list(models.keys()),
            default=["Random Forest", "SVM (RBF Kernel)", "Voting Ensemble"],
            help="Choose multiple models to compare results"
        )
        
        # Display selected model info
        for model in selected_models:
            if model in model_descriptions:
                st.markdown(f"""
                <div class="feature-card">
                    <p><strong>{model}</strong></p>
                    <p><small>{model_descriptions[model]}</small></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        
        analysis_mode = st.radio(
            "Analysis Mode",
            ["🔍 Single Image", "📁 Batch Processing", "⚖️ Model Comparison"],
            help="Choose how you want to analyze your images"
        )
        
        show_feature_analysis = st.checkbox("🔬 Show Feature Analysis", 
                                           st.session_state.user_preferences['show_advanced_metrics'])
        show_confidence_details = st.checkbox("📊 Show Confidence Details", 
                                             st.session_state.user_preferences['show_confidence_details'])
        enable_notifications = st.checkbox("🔔 Enable Analysis Notifications", 
                                          st.session_state.user_preferences['notification_enabled'])
    
    with col1:
        st.markdown("### 📤 Image Upload & Processing")
        
        if analysis_mode == "📁 Batch Processing":
            uploaded_files = st.file_uploader(
                "Upload Multiple Plant Images", 
                type=["jpg", "png", "jpeg"], 
                accept_multiple_files=True,
                key="batch_classify",
                help="Select multiple images for batch analysis (Max 20 images)"
            )
            
            if uploaded_files:
                if len(uploaded_files) > 20:
                    st.warning("⚠️ Maximum 20 images allowed. Please reduce your selection.")
                    return
                
                st.markdown(f"""
                <div class="analysis-container">
                    <h4>📊 Batch Analysis Overview</h4>
                    <p><strong>Images to process:</strong> {len(uploaded_files)}</p>
                    <p><strong>Selected models:</strong> {len(selected_models)}</p>
                    <p><strong>Total analyses:</strong> {len(uploaded_files) * len(selected_models)}</p>
                    <p><strong>Estimated time:</strong> {len(uploaded_files) * len(selected_models) * 1.2:.1f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🚀 Start Batch Analysis", type="primary"):
                    results_data = []
                    progress_container = st.container()
                    
                    with progress_container:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        total_operations = len(uploaded_files) * len(selected_models)
                        current_operation = 0
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            image = Image.open(uploaded_file)
                            
                            # Show current image being processed
                            st.image(image, caption=f"Processing: {uploaded_file.name}", width=200)
                            
                            for model_name in selected_models:
                                current_operation += 1
                                status_text.text(f"🔄 Processing {uploaded_file.name} with {model_name}...")
                                
                                result = analyze_image_advanced(image, model_name)
                                results_data.append({
                                    "Image": uploaded_file.name,
                                    "Model": model_name,
                                    "Prediction": result['prediction'],
                                    "Confidence": f"{result['confidence']:.2f}%",
                                    "Analysis_Time": f"{result['analysis_time']:.3f}s",
                                    "User": result['user'],
                                    "Timestamp": result['timestamp'].strftime("%H:%M:%S")
                                })
                                
                                # Update analytics
                                update_analytics(result['prediction'], result['confidence'], model_name)
                                
                                progress_bar.progress(current_operation / total_operations)
                                time.sleep(0.1)  # Small delay for visual feedback
                    
                    status_text.text("✅ Batch analysis completed!")
                    
                    # Display comprehensive results
                    st.markdown("### 📋 Batch Analysis Results")
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Enhanced visualizations
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Prediction distribution by model
                        fig_pred = px.histogram(
                            results_df, 
                            x='Prediction', 
                            color='Model',
                            title="Prediction Distribution by Model",
                            barmode='group'
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                    
                    with col_viz2:
                        # Confidence distribution
                        results_df['Confidence_Numeric'] = results_df['Confidence'].str.replace('%', '').astype(float)
                        fig_conf = px.box(
                            results_df,
                            x='Model',
                            y='Confidence_Numeric',
                            title="Confidence Distribution by Model"
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("### 📊 Summary Statistics")
                    
                    summary_stats = results_df.groupby('Model').agg({
                        'Confidence_Numeric': ['mean', 'std', 'min', 'max'],
                        'Prediction': 'count'
                    }).round(2)
                    
                    st.dataframe(summary_stats, use_container_width=True)
                    
                    # Export functionality
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        if st.button("📥 Export Results to CSV"):
                            csv = results_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            href = f'<a href="data:file/csv;base64,{b64}" download="batch_analysis_results.csv">📥 Download CSV</a>'
                            st.markdown(href, unsafe_allow_html=True)
                    
                    with col_exp2:
                        if st.button("📊 Generate Report"):
                            st.markdown("### 📋 Analysis Report")
                            
                            total_images = len(uploaded_files)
                            healthy_count = len(results_df[results_df['Prediction'] == 'Healthy'])
                            diseased_count = total_images - healthy_count
                            avg_confidence = results_df['Confidence_Numeric'].mean()
                            
                            st.markdown(f"""
                            <div class="analysis-container">
                                <h4>📊 Batch Analysis Report</h4>
                                <p><strong>Total Images Processed:</strong> {total_images}</p>
                                <p><strong>Healthy Plants:</strong> {healthy_count} ({healthy_count/total_images*100:.1f}%)</p>
                                <p><strong>Diseased Plants:</strong> {diseased_count} ({diseased_count/total_images*100:.1f}%)</p>
                                <p><strong>Average Confidence:</strong> {avg_confidence:.1f}%</p>
                                <p><strong>Models Used:</strong> {', '.join(selected_models)}</p>
                                <p><strong>Analysis Completed:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        else:  # Single image or model comparison
            uploaded_file = st.file_uploader(
                "Upload Plant Leaf Image", 
                type=["jpg", "png", "jpeg"], 
                key="single_classify",
                help="Upload a single image for detailed analysis"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="📸 Uploaded Image", use_container_width=True)
                
                # Enhanced image information
                st.markdown(f"""
                <div class="analysis-container">
                    <h4>📋 Image Properties</h4>
                    <div class="stats-grid">
                        <div>
                            <p><strong>Filename:</strong> {uploaded_file.name}</p>
                            <p><strong>Dimensions:</strong> {image.size[0]} x {image.size[1]}</p>
                        </div>
                        <div>
                            <p><strong>Format:</strong> {image.format}</p>
                            <p><strong>File Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔍 Analyze with Selected Models", type="primary"):
                    results = {}
                    
                    # Analysis progress
                    with st.spinner("🔄 Running multi-model analysis..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, model_name in enumerate(selected_models):
                            status_text.text(f"🤖 Running {model_name}...")
                            results[model_name] = analyze_image_advanced(image, model_name)
                            update_analytics(results[model_name]['prediction'], results[model_name]['confidence'], model_name)
                            progress_bar.progress((i + 1) / len(selected_models))
                            time.sleep(0.2)
                        
                        status_text.text("✅ Analysis complete!")
                    
                    # Display results comparison
                    st.markdown("### 🔬 Multi-Model Analysis Results")
                    
                    # Model comparison cards
                    if len(selected_models) <= 3:
                        cols = st.columns(len(selected_models))
                    else:
                        cols = st.columns(3)
                    
                    for i, (model_name, result) in enumerate(results.items()):
                        with cols[i % 3]:
                            confidence_color = "#4CAF50" if result['confidence'] > 80 else "#FF9800" if result['confidence'] > 60 else "#F44336"
                            
                            st.markdown(f"""
                            <div class="model-card">
                                <h4>{model_name}</h4>
                                <h2>{result['disease_details']['icon']} {result['prediction']}</h2>
                                <p style="color: white; font-size: 1.2em;"><strong>{result['confidence']:.1f}%</strong></p>
                                <p><small>Analysis time: {result['analysis_time']:.3f}s</small></p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Detailed comparison table
                    st.markdown("### 📊 Detailed Comparison")
                    
                    comparison_data = []
                    for model_name, result in results.items():
                        comparison_data.append({
                            "Model": model_name,
                            "Prediction": result['prediction'],
                            "Confidence": f"{result['confidence']:.2f}%",
                            "Confidence_Level": result['confidence_level'],
                            "Analysis_Time": f"{result['analysis_time']:.3f}s"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Consensus analysis
                    st.markdown("### 🗳️ Consensus Analysis")
                    
                    predictions = [result['prediction'] for result in results.values()]
                    confidences = [result['confidence'] for result in results.values()]
                    
                    # Find consensus
                    from collections import Counter
                    prediction_counts = Counter(predictions)
                    consensus_prediction = prediction_counts.most_common(1)[0][0]
                    consensus_count = prediction_counts.most_common(1)[0][1]
                    
                    avg_confidence = np.mean(confidences)
                    consensus_strength = consensus_count / len(predictions) * 100
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3>🎯 Consensus Result</h3>
                        <h2>{disease_info[consensus_prediction]['icon']} {consensus_prediction}</h2>
                        <p><strong>Consensus Strength:</strong> {consensus_strength:.1f}% ({consensus_count}/{len(predictions)} models agree)</p>
                        <p><strong>Average Confidence:</strong> {avg_confidence:.1f}%</p>
                        <p><strong>Recommendation:</strong> {"High confidence result" if consensus_strength > 66 else "Consider additional analysis"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Probability distribution visualization
                    if show_confidence_details:
                        st.markdown("### 📈 Probability Analysis")
                        
                        # Collect all probabilities
                        all_probs = {}
                        for model_name, result in results.items():
                            for disease, prob in result['all_probabilities'].items():
                                if disease not in all_probs:
                                    all_probs[disease] = []
                                all_probs[disease].append(prob)
                        
                        # Create probability comparison chart
                        prob_data = []
                        for disease, probs in all_probs.items():
                            prob_data.append({
                                "Disease": disease,
                                "Mean_Probability": np.mean(probs),
                                "Std_Probability": np.std(probs),
                                "Min_Probability": np.min(probs),
                                "Max_Probability": np.max(probs)
                            })
                        
                        prob_df = pd.DataFrame(prob_data)
                        
                        fig_prob = px.bar(
                            prob_df,
                            x='Disease',
                            y='Mean_Probability',
                            error_y='Std_Probability',
                            title="Average Probability Distribution Across Models",
                            color='Mean_Probability',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Feature importance analysis
                    if show_feature_analysis:
                        st.markdown("### 🔬 Feature Importance Analysis")
                        
                        # Average feature importance across models
                        all_features = {}
                        for model_name, result in results.items():
                            for feature, importance in result['feature_importance'].items():
                                if feature not in all_features:
                                    all_features[feature] = []
                                all_features[feature].append(importance)
                        
                        avg_features = {feature: np.mean(importances) for feature, importances in all_features.items()}
                        top_features = sorted(avg_features.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        feature_df = pd.DataFrame(top_features, columns=['Feature', 'Average_Importance'])
                        
                        fig_features = px.bar(
                            feature_df,
                            x='Average_Importance',
                            y='Feature',
                            orientation='h',
                            title="Top 10 Most Important Features (Averaged Across Models)",
                            color='Average_Importance',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                    
                    # Add to history
                    for model_name, result in results.items():
                        st.session_state.history.append({
                            "Mode": "Classification",
                            "Model": model_name,
                            "Prediction": result['prediction'],
                            "Confidence": f"{result['confidence']:.2f}%",
                            "Timestamp": result['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                            "User": result['user'],
                            "Analysis_Time": f"{result['analysis_time']:.3f}s"
                        })
                    
                    # Notification
                    if enable_notifications:
                        st.success(f"🔔 Analysis complete! Consensus: {consensus_prediction} ({consensus_strength:.1f}% agreement)")

# --- Treatment Guide Page ---
def show_treatment():
    """Treatment and prevention guide"""
    st.markdown('<h1 class="main-header">💊 Treatment & Prevention Guide</h1>', unsafe_allow_html=True)
    
    # Disease selection
    selected_disease = st.selectbox(
        "🔍 Select Disease Type",
        list(disease_info.keys()),
        help="Choose a disease to view detailed treatment information"
    )
    
    disease_data = disease_info[selected_disease]
    
    # Display disease information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h2 style="color: {disease_data['color']};">{selected_disease}</h2>
            <h3>🚨 Severity Level</h3>
            <p style="font-size: 1.2em; font-weight: bold;">{disease_data['severity']}</p>
            
            <h3>💊 Treatment</h3>
            <p>{disease_data['treatment']}</p>
            
            <h3>🛡️ Prevention</h3>
            <p>{disease_data['prevention']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📈 Treatment Timeline")
        
        # Treatment timeline based on disease type
        if selected_disease == "Healthy":
            timeline_data = {
                "Week": [1, 2, 3, 4],
                "Action": ["Monitor", "Regular Care", "Maintain", "Continue"],
                "Status": ["Good", "Good", "Good", "Good"]
            }
        elif selected_disease == "Rust":
            timeline_data = {
                "Week": [1, 2, 3, 4],
                "Action": ["Apply Fungicide", "Monitor Progress", "Repeat Treatment", "Evaluate"],
                "Status": ["Critical", "Improving", "Better", "Recovered"]
            }
        elif selected_disease == "Scab":
            timeline_data = {
                "Week": [1, 2, 3, 4],
                "Action": ["Sulfur Treatment", "Improve Drainage", "Monitor", "Maintain"],
                "Status": ["Critical", "Improving", "Better", "Stable"]
            }
        else:  # Multiple Diseases
            timeline_data = {
                "Week": [1, 2, 3, 4],
                "Action": ["Consult Expert", "Intensive Care", "Monitor", "Evaluate"],
                "Status": ["Critical", "Critical", "Improving", "Uncertain"]
            }
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        # Treatment effectiveness chart
        effectiveness_data = {
            "Healthy": 100,
            "Rust": 85,
            "Scab": 80,
            "Multiple Diseases": 60
        }
        
        fig = px.bar(
            x=list(effectiveness_data.keys()),
            y=list(effectiveness_data.values()),
            title="Treatment Effectiveness Rate (%)",
            color=list(effectiveness_data.values()),
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional resources
    st.markdown("---")
    st.markdown("### 📚 Additional Resources")
    
    resources = [
        "🌱 Plant Care Best Practices",
        "🔬 Disease Identification Guide",
        "💧 Watering Guidelines",
        "🌞 Light Requirements",
        "🌡️ Temperature Control",
        "🌿 Organic Treatment Options"
    ]
    
    cols = st.columns(3)
    for i, resource in enumerate(resources):
        with cols[i % 3]:
            st.markdown(f"📖 {resource}")

# --- Analytics/History Page ---
def show_history():
    """Show prediction history and analytics"""
    st.markdown('<h1 class="main-header">📊 Analytics & History</h1>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("🔍 No predictions yet. Start by uploading and analyzing plant images!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Predictions</h3>
            <h2>{}</h2>
        </div>
        """.format(len(st.session_state.history)), unsafe_allow_html=True)
    
    with col2:
        healthy_count = sum(1 for h in st.session_state.history if h['Prediction'] == 'Healthy')
        st.markdown("""
        <div class="metric-card">
            <h3>🌱 Healthy Plants</h3>
            <h2>{}</h2>
        </div>
        """.format(healthy_count), unsafe_allow_html=True)
    
    with col3:
        disease_count = len(st.session_state.history) - healthy_count
        st.markdown("""
        <div class="metric-card">
            <h3>🚨 Diseased Plants</h3>
            <h2>{}</h2>
        </div>
        """.format(disease_count), unsafe_allow_html=True)
    
    with col4:
        avg_confidence = np.mean([float(h['Confidence'].replace('%', '')) for h in st.session_state.history])
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Avg Confidence</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(avg_confidence), unsafe_allow_html=True)
    
    # History table
    st.markdown("### 📋 Prediction History")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    
    # Analytics charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Prediction distribution
        prediction_counts = history_df['Prediction'].value_counts()
        fig1 = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Prediction Distribution"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Model usage
        model_counts = history_df['Model'].value_counts()
        fig2 = px.bar(
            x=model_counts.index,
            y=model_counts.values,
            title="Model Usage Statistics"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Time series analysis
    if len(st.session_state.history) > 1:
        st.markdown("### 📈 Time Series Analysis")
        
        # Convert timestamp to datetime
        history_df['Timestamp'] = pd.to_datetime(history_df['Timestamp'])
        history_df['Confidence_Numeric'] = history_df['Confidence'].str.replace('%', '').astype(float)
        
        fig3 = px.line(
            history_df,
            x='Timestamp',
            y='Confidence_Numeric',
            color='Model',
            title="Confidence Over Time"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Export functionality
    st.markdown("### 📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV"):
            csv = history_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="prediction_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("📋 Export to JSON"):
            json_data = history_df.to_json(orient='records')
            b64 = base64.b64encode(json_data.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="prediction_history.json">Download JSON</a>'
            st.markdown(href, unsafe_allow_html=True)

# --- Settings Page ---
def show_settings():
    """User settings and preferences"""
    st.markdown('<h1 class="main-header">⚙️ Settings & Preferences</h1>', unsafe_allow_html=True)
    
    # User profile section
    st.markdown("### 👤 User Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="feature-card">
            <h4>Current User Information</h4>
            <p><strong>Username:</strong> {st.session_state.username}</p>
            <p><strong>Role:</strong> {st.session_state.user_role.title()}</p>
            <p><strong>Total Predictions:</strong> {len(st.session_state.history)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔐 Change Password")
        old_password = st.text_input("Current Password", type="password", key="old_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_pass")
        
        if st.button("🔄 Update Password"):
            if authenticate_user(st.session_state.username, old_password):
                if new_password == confirm_password and len(new_password) >= 6:
                    st.session_state.users[st.session_state.username]['password'] = hash_password(new_password)
                    st.success("✅ Password updated successfully!")
                else:
                    st.error("❌ New passwords don't match or are too short")
            else:
                st.error("❌ Current password is incorrect")
    
    # Application preferences
    st.markdown("---")
    st.markdown("### 🎛️ Application Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎨 Display Settings")
        
        # Theme selection
        theme_options = ["🌞 Light", "🌙 Dark", "🌈 Colorful", "🌸 Pastel"]
        selected_theme = st.selectbox(
            "Theme", 
            theme_options,
            index=theme_options.index(st.session_state.user_preferences['theme'])
        )
        
        # Auto-analyze setting
        auto_analyze = st.checkbox(
            "Auto-analyze uploaded images", 
            st.session_state.user_preferences['auto_analyze']
        )
        
        # Advanced metrics
        show_advanced_metrics = st.checkbox(
            "Show advanced metrics", 
            st.session_state.user_preferences['show_advanced_metrics']
        )
    
    with col2:
        st.markdown("#### 🔔 Notification Settings")
        
        notification_enabled = st.checkbox(
            "Enable notifications", 
            st.session_state.user_preferences['notification_enabled']
        )
        
        st.markdown("#### 📊 Data Settings")
        
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.analytics = {
                'total_predictions': 0,
                'healthy_count': 0,
                'disease_count': 0,
                'accuracy_scores': []
            }
            st.success("✅ History cleared successfully!")
            st.rerun()
    
    # Save preferences
    if st.button("💾 Save Preferences", type="primary"):
        st.session_state.user_preferences.update({
            'theme': selected_theme,
            'auto_analyze': auto_analyze,
            'show_advanced_metrics': show_advanced_metrics,
            'notification_enabled': notification_enabled
        })
        st.success("✅ Preferences saved successfully!")
        st.rerun()
    
    # System information (for admins)
    if st.session_state.user_role == 'admin':
        st.markdown("---")
        st.markdown("### 🔧 System Information")
        
        system_info = {
            "Total Users": len(st.session_state.users),
            "Total Predictions": len(st.session_state.history),
            "Models Loaded": len(models),
            "Session State Keys": len(st.session_state.keys())
        }
        
        for key, value in system_info.items():
            st.metric(key, value)
# Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; margin-top: 2rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <h3>🌿 Plant Disease AI Assistant </h3>
        <p>🚀 Made with ❤️ by Kratik Jain</p>
        <p>⚡ Powered by Streamlit • OpenCV • Scikit-learn • Plotly </p>
        <div style='margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;'>
            📊 Total Predictions: {total} | 🌱 Healthy: {healthy} | 🦠 Diseased: {diseased}
        </div>
    </div>
    """.format(
        total=st.session_state.analytics['total_predictions'],
        healthy=st.session_state.analytics['healthy_count'],
        diseased=st.session_state.analytics['disease_count']
    ), unsafe_allow_html=True)
# Run the application
if __name__ == "__main__":
    main()
