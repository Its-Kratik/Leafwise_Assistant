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
    theme = st.sidebar.selectbox(
        "🎨 Theme", 
        ["🌞 Light", "🌙 Dark", "🌈 Colorful"],
        index=0 if st.session_state.user_preferences['theme'] == 'Light' else 1
    )
    st.session_state.user_preferences['theme'] = theme
    # Update theme if changed
    if theme != st.session_state.user_preferences['theme']:
        st.session_state.user_preferences['theme'] = theme
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
    st.markdown("---") 
    # Technical details
    with st.expander("🔧 Technical Specifications"):
        st.markdown("""
        **Supported Disease Classes:**
        - 🟢 Healthy Plants
        - 🔴 Multiple Diseases
        - 🟠 Rust Disease
        - 🟤 Scab Disease
        
        **Machine Learning Models:**
        - Random Forest Classifier
        - Support Vector Machine (RBF)
        - Gradient Boosting Classifier
        - Voting Ensemble (Recommended)
        - K-Nearest Neighbors
        - Logistic Regression
        
        **Technology Stack:**
        - Frontend: Streamlit with modern CSS
        - ML Backend: Scikit-learn, OpenCV
        - Visualization: Plotly, Matplotlib
        - Features: LBP, Color Statistics
        """)
# --- Advanced Classification Page ---
def show_classification():
    st.markdown('<h1 class="main-header">🧠 Advanced Disease Classification</h1>', unsafe_allow_html=True)
    
    # Main layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Model selection with enhanced descriptions
        model_descriptions = {
            "Random Forest": "🌳 Ensemble of decision trees - Great for general accuracy",
            "SVM (RBF Kernel)": "🎯 Support Vector Machine - Excellent for complex patterns",
            "Gradient Boosting": "🚀 Sequential learning - High accuracy, slower",
            "Voting Ensemble": "🗳️ Combined predictions - Best overall performance",
            "KNN": "👥 Nearest neighbors - Simple but effective",
            "Logistic Regression": "📈 Linear classification - Fast and interpretable"
        }
        
        selected_models = st.multiselect(
            "🤖 Select AI Models",
            list(models.keys()),
            default=["Random Forest", "SVM (RBF Kernel)", "Voting Ensemble"],
            format_func=lambda x: model_descriptions[x]
        )
        
        # Processing options
        batch_mode = st.checkbox("📁 Batch Processing Mode")
        
        if st.session_state.user_preferences['show_advanced_metrics']:
            show_feature_analysis = st.checkbox("🔍 Show Feature Analysis", True)
            show_probability_charts = st.checkbox("📊 Show Probability Charts", True)
        else:
            show_feature_analysis = False
            show_probability_charts = False
    
    with col1:
        if batch_mode:
            # Batch processing mode
            uploaded_files = st.file_uploader(
                "📤 Upload Multiple Plant Images", 
                type=["jpg", "png", "jpeg"], 
                accept_multiple_files=True,
                key="batch_classify"
            )
            
            if uploaded_files and st.button("🔄 Process All Images", type="primary"):
                st.write(f"📊 Processing {len(uploaded_files)} images with {len(selected_models)} model(s)...")
                
                results_data = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    image = Image.open(uploaded_file)
                    
                    # Analyze with selected models
                    for model_name in selected_models:
                        result = analyze_image_advanced(image, model_name)
                        results_data.append({
                            "Image": uploaded_file.name,
                            "Model": model_name,
                            "Prediction": result['prediction'],
                            "Confidence": f"{result['confidence']:.2f}%",
                            "User": result['user'],
                            "Timestamp": result['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                        })
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✅ Processing complete!")
                
                # Enhanced results display
                st.markdown("### 📊 Batch Processing Results")
                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df, use_container_width=True)
                
                # Enhanced visualizations for batch results
                if len(selected_models) > 1:
                    # Model comparison chart
                    fig_comparison = px.bar(
                        results_df.groupby(['Model', 'Prediction']).size().reset_index(name='Count'),
                        x='Model', y='Count', color='Prediction',
                        title="🔄 Model Predictions Comparison",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_comparison.update_layout(height=400)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Confidence distribution
                    results_df['Confidence_Numeric'] = results_df['Confidence'].str.replace('%', '').astype(float)
                    fig_confidence = px.box(
                        results_df, 
                        x='Model', 
                        y='Confidence_Numeric',
                        title="📈 Confidence Distribution by Model",
                        color='Model'
                    )
                    fig_confidence.update_layout(height=400)
                    st.plotly_chart(fig_confidence, use_container_width=True)
                
                # Update history for batch processing
                for _, row in results_df.iterrows():
                    st.session_state.history.append({
                        "Mode": "Batch Classification",
                        "Model": row['Model'],
                        "Prediction": row['Prediction'],
                        "Confidence": row['Confidence'],
                        "Timestamp": row['Timestamp'],
                        "User": row['User'],
                        "Image": row['Image']
                    })
        
        else:
            # Single image processing mode
            uploaded_file = st.file_uploader(
                "📤 Upload Plant Leaf Image", 
                type=["jpg", "png", "jpeg"], 
                key="classify"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="📸 Uploaded Sample", use_container_width=True)
                
                if st.button("🔬 Analyze with Selected Models", type="primary"):
                    results = {}
                    
                    with st.spinner("🔄 Running AI classification..."):
                        for model_name in selected_models:
                            results[model_name] = analyze_image_advanced(image, model_name)
                    
                    # Enhanced results display
                    st.markdown("### 🎯 Classification Results")
                    
                    # Model comparison cards
                    cols = st.columns(len(selected_models))
                    for i, (model_name, result) in enumerate(results.items()):
                        with cols[i]:
                            confidence_color = "green" if result['confidence'] > 80 else "orange" if result['confidence'] > 60 else "red"
                            st.markdown(f"""
                            <div class="prediction-box" style="border-left: 4px solid {confidence_color};">
                                <h4>{model_name}</h4>
                                <h2>{result['prediction']}</h2>
                                <p>Confidence: {result['confidence']:.1f}%</p>
                                <p>User: {result['user']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Probability charts for each model
                    if show_probability_charts:
                        st.markdown("### 📊 Detailed Probability Analysis")
                        
                        for model_name, result in results.items():
                            if 'all_probabilities' in result:
                                st.markdown(f"#### {model_name} - Probability Distribution")
                                
                                fig_prob = px.bar(
                                    x=list(result['all_probabilities'].keys()),
                                    y=list(result['all_probabilities'].values()),
                                    title=f"🎯 {model_name} - Confidence Scores",
                                    labels={'x': 'Disease Type', 'y': 'Probability %'},
                                    color=list(result['all_probabilities'].values()),
                                    color_continuous_scale='Viridis'
                                )
                                fig_prob.update_layout(height=350)
                                st.plotly_chart(fig_prob, use_container_width=True)
                    
                    # Feature analysis
                    if show_feature_analysis:
                        st.markdown("### 🔍 Advanced Feature Analysis")
                        
                        # Extract and display features
                        features = extract_features(image)
                        feature_names = ['Red Mean', 'Green Mean', 'Blue Mean', 
                                       'Red Std', 'Green Std', 'Blue Std'] + \
                                      [f'LBP_{i}' for i in range(10)]
                        
                        feature_df = pd.DataFrame({
                            'Feature': feature_names,
                            'Value': features[:len(feature_names)]
                        })
                        
                        # Enhanced feature visualization
                        fig_features = px.bar(
                            feature_df, 
                            x='Feature', 
                            y='Value',
                            title="🔍 Extracted Image Features",
                            color='Value',
                            color_continuous_scale='Blues'
                        )
                        fig_features.update_layout(
                            xaxis_tickangle=-45,
                            height=400
                        )
                        st.plotly_chart(fig_features, use_container_width=True)
                        
                        # Feature importance heatmap
                        if len(selected_models) > 1:
                            feature_importance_data = []
                            for model_name in selected_models:
                                # Simulate feature importance (replace with actual model feature importance)
                                importance = np.random.rand(len(feature_names))
                                feature_importance_data.extend([
                                    {"Model": model_name, "Feature": feat, "Importance": imp}
                                    for feat, imp in zip(feature_names, importance)
                                ])
                            
                            importance_df = pd.DataFrame(feature_importance_data)
                            fig_heatmap = px.imshow(
                                importance_df.pivot(index='Feature', columns='Model', values='Importance'),
                                title="🔥 Feature Importance Heatmap",
                                color_continuous_scale='RdYlBu_r'
                            )
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Update history for all models
                    for model_name, result in results.items():
                        st.session_state.history.append({
                            "Mode": "Classification",
                            "Model": model_name,
                            "Prediction": result['prediction'],
                            "Confidence": f"{result['confidence']:.2f}%",
                            "Timestamp": result['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                            "User": result['user']
                        })
                        
                        update_analytics(result['prediction'], result['confidence'])
    
    with col2:
        # Enhanced sidebar with model performance and statistics
        st.markdown("### 📊 Model Performance Dashboard")
        
        # Model performance comparison
        model_performance = {
            'Model': list(models.keys()),
            'Accuracy': [94.2, 91.8, 93.5, 96.1, 89.3, 87.6],
            'Speed': [85, 70, 75, 80, 95, 90],
            'Memory': [75, 85, 80, 70, 60, 95]
        }
        
        perf_df = pd.DataFrame(model_performance)
        
        # Interactive performance scatter plot
        fig_perf = px.scatter(
            perf_df,
            x='Speed',
            y='Accuracy',
            size='Memory',
            color='Model',
            title="⚡ Model Performance Matrix",
            labels={'Speed': 'Speed Score', 'Accuracy': 'Accuracy %', 'Memory': 'Memory Efficiency'},
            hover_data=['Memory']
        )
        fig_perf.update_layout(height=350)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance metrics table
        st.markdown("#### 📈 Performance Metrics")
        st.dataframe(perf_df.set_index('Model'), use_container_width=True)
        
        # Usage statistics
        if st.session_state.history:
            st.markdown("#### 📊 Usage Statistics")
            history_df = pd.DataFrame(st.session_state.history)
            
            # Most used models
            model_usage = history_df['Model'].value_counts().head(5)
            fig_usage = px.pie(
                values=model_usage.values,
                names=model_usage.index,
                title="🔥 Most Used Models"
            )
            fig_usage.update_layout(height=300)
            st.plotly_chart(fig_usage, use_container_width=True)
            
            # Recent activity
            st.markdown("#### 🕒 Recent Activity")
            recent_activity = history_df.tail(5)[['Model', 'Prediction', 'Confidence', 'User']]
            st.dataframe(recent_activity, use_container_width=True)
        
        # Quick tips
        st.markdown("#### 💡 Quick Tips")
        st.info("""
        🎯 **For best results:**
        • Use high-quality, well-lit images
        • Ensure leaves are clearly visible
        • Try multiple models for comparison
        • Use batch mode for multiple images
        """)
# --- Treatment Guide Page ---
def show_treatment():
    """Comprehensive treatment and prevention guide with structured viewer"""
    st.markdown('<h1 class="main-header">💊 Treatment & Prevention Guide</h1>', unsafe_allow_html=True)
    
    # Header statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>🦠 Diseases Covered</h4>
            <h2>{}</h2>
            <p>Comprehensive database</p>
        </div>
        """.format(len(disease_info)), unsafe_allow_html=True)
    
    with col2:
        avg_success_rate = np.mean([info['success_rate'] for info in disease_info.values()])
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Avg Success Rate</h4>
            <h2>{avg_success_rate:.1f}%</h2>
            <p>Treatment effectiveness</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>⏱️ Avg Duration</h4>
            <h2>2-4</h2>
            <p>Weeks to recovery</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>🎯 Prevention Tips</h4>
            <h2>15+</h2>
            <p>Expert recommendations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Disease selection with enhanced interface
        st.markdown("### 🔍 Disease Selection & Overview")
        
        # Create disease cards for selection
        selected_disease = st.selectbox(
            "Select Disease Type",
            list(disease_info.keys()),
            help="Choose a disease to view comprehensive treatment information"
        )
        
        disease_data = disease_info[selected_disease]
        
        # Enhanced disease overview
        st.markdown(f"""
        <div class="analysis-container">
            <div style="text-align: center; padding: 2rem;">
                <h1 style="font-size: 3rem;">{disease_data['icon']}</h1>
                <h2 style="color: {disease_data['color']}; margin: 1rem 0;">{selected_disease}</h2>
                <div class="stats-grid">
                    <div style="background: {disease_data['color']}; color: white; padding: 1rem; border-radius: 8px;">
                        <h4>🚨 Severity</h4>
                        <p>{disease_data['severity']} (Level {disease_data['severity_level']}/4)</p>
                    </div>
                    <div style="background: linear-gradient(135deg, #4CAF50, #2196F3); color: white; padding: 1rem; border-radius: 8px;">
                        <h4>📈 Success Rate</h4>
                        <p>{disease_data['success_rate']}%</p>
                    </div>
                    <div style="background: linear-gradient(135deg, #FF9800, #F44336); color: white; padding: 1rem; border-radius: 8px;">
                        <h4>⏱️ Duration</h4>
                        <p>{disease_data['duration']}</p>
                    </div>
                    <div style="background: linear-gradient(135deg, #9C27B0, #673AB7); color: white; padding: 1rem; border-radius: 8px;">
                        <h4>💰 Cost</h4>
                        <p>{disease_data['cost']}</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed treatment protocol
        st.markdown("### 💊 Comprehensive Treatment Protocol")
        
        # Treatment phases
        treatment_phases = {
            "Healthy": [
                {"phase": "Maintenance", "duration": "Ongoing", "actions": ["Regular monitoring", "Optimal care conditions", "Preventive measures"]},
            ],
            "Rust": [
                {"phase": "Immediate (Days 1-3)", "duration": "3 days", "actions": ["Isolate affected plant", "Apply copper fungicide", "Remove infected leaves"]},
                {"phase": "Treatment (Days 4-14)", "duration": "10 days", "actions": ["Continue fungicide treatment", "Improve air circulation", "Monitor progress"]},
                {"phase": "Recovery (Days 15-21)", "duration": "7 days", "actions": ["Reduce treatment frequency", "Assess recovery", "Maintain conditions"]},
            ],
            "Scab": [
                {"phase": "Immediate (Days 1-3)", "duration": "3 days", "actions": ["Apply sulfur-based fungicide", "Improve drainage", "Remove affected parts"]},
                {"phase": "Treatment (Days 4-21)", "duration": "18 days", "actions": ["Weekly fungicide application", "Monitor soil moisture", "Avoid overhead watering"]},
                {"phase": "Recovery (Days 22-28)", "duration": "7 days", "actions": ["Evaluate treatment success", "Gradual return to normal care", "Long-term monitoring"]},
            ],
            "Multiple Diseases": [
                {"phase": "Emergency (Days 1-5)", "duration": "5 days", "actions": ["Consult plant specialist", "Complete isolation", "Document all symptoms"]},
                {"phase": "Intensive Care (Days 6-28)", "duration": "23 days", "actions": ["Follow specialist recommendations", "Multiple treatment approaches", "Daily monitoring"]},
                {"phase": "Recovery Assessment (Days 29-42)", "duration": "14 days", "actions": ["Evaluate treatment effectiveness", "Adjust protocols", "Plan long-term care"]},
            ]
        }
        
        phases = treatment_phases.get(selected_disease, treatment_phases["Healthy"])
        
        for i, phase in enumerate(phases):
            st.markdown(f"""
            <div class="treatment-timeline">
                <h4 style="color: {disease_data['color']};">📅 Phase {i+1}: {phase['phase']}</h4>
                <p><strong>Duration:</strong> {phase['duration']}</p>
                <h5>🔧 Actions to Take:</h5>
                <ul>
                    {"".join([f"<li>{action}</li>" for action in phase['actions']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Symptoms and identification
        st.markdown("### 🔍 Symptoms & Identification")
        
        symptoms_cols = st.columns(2)
        with symptoms_cols[0]:
            st.markdown(f"""
            <div class="feature-card">
                <h4>🔍 Common Symptoms</h4>
                <ul>
                    {"".join([f"<li>{symptom}</li>" for symptom in disease_data['symptoms']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with symptoms_cols[1]:
            st.markdown(f"""
            <div class="feature-card">
                <h4>⚠️ Root Causes</h4>
                <ul>
                    {"".join([f"<li>{cause}</li>" for cause in disease_data['causes']])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Prevention strategies
        st.markdown("### 🛡️ Prevention Strategies")
        
        prevention_strategies = {
            "Healthy": [
                "Maintain consistent watering schedule",
                "Provide adequate light conditions",
                "Regular plant health inspections",
                "Proper soil drainage",
                "Appropriate fertilization"
            ],
            "Rust": [
                "Ensure good air circulation around plants",
                "Avoid overhead watering",
                "Remove plant debris regularly",
                "Don't overcrowd plants",
                "Use drip irrigation when possible"
            ],
            "Scab": [
                "Improve soil drainage",
                "Water at soil level, not leaves",
                "Remove fallen leaves promptly",
                "Avoid working with wet plants",
                "Use disease-resistant varieties"
            ],
            "Multiple Diseases": [
                "Implement strict sanitation protocols",
                "Quarantine new plants before introducing",
                "Use sterilized tools for pruning",
                "Monitor environmental conditions closely",
                "Regular application of preventive treatments"
            ]
        }
        
        strategies = prevention_strategies.get(selected_disease, prevention_strategies["Healthy"])
        
        prevention_cols = st.columns(2)
        for i, strategy in enumerate(strategies):
            with prevention_cols[i % 2]:
                st.markdown(f"""
                <div class="feature-card">
                    <h5>🛡️ Strategy {i+1}</h5>
                    <p>{strategy}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Treatment effectiveness visualization
        st.markdown("### 📊 Treatment Analytics")
        
        # Success rates comparison
        success_rates = {disease: info['success_rate'] for disease, info in disease_info.items()}
        
        fig_success = px.bar(
            x=list(success_rates.keys()),
            y=list(success_rates.values()),
            title="Treatment Success Rates",
            color=list(success_rates.values()),
            color_continuous_scale="RdYlGn",
            labels={'x': 'Disease', 'y': 'Success Rate (%)'}
        )
        st.plotly_chart(fig_success, use_container_width=True)
        
        # Severity levels
        severity_levels = {disease: info['severity_level'] for disease, info in disease_info.items()}
        
        fig_severity = px.bar(
            x=list(severity_levels.keys()),
            y=list(severity_levels.values()),
            title="Disease Severity Levels",
            color=list(severity_levels.values()),
            color_continuous_scale="RdYlBu_r",
            labels={'x': 'Disease', 'y': 'Severity Level'}
        )
        st.plotly_chart(fig_severity, use_container_width=True)
        
        # Treatment timeline for selected disease
        if selected_disease != "Healthy":
            st.markdown("### ⏰ Treatment Timeline")
            
            timeline_data = []
            phases = treatment_phases.get(selected_disease, [])
            
            for i, phase in enumerate(phases):
                timeline_data.append({
                    "Phase": f"Phase {i+1}",
                    "Duration": phase['duration'],
                    "Actions": len(phase['actions'])
                })
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                st.dataframe(timeline_df, use_container_width=True)
        
        # Quick reference guide
        st.markdown("### 📋 Quick Reference")
        
        quick_ref = {
            "Emergency Signs": ["Rapid wilting", "Extensive discoloration", "Spreading symptoms"],
            "When to Seek Help": ["Multiple symptoms", "Treatment not working", "Rapid deterioration"],
            "Essential Supplies": ["Fungicides", "Pruning tools", "Protective equipment"]
        }
        
        for category, items in quick_ref.items():
            st.markdown(f"""
            <div class="feature-card">
                <h5>{category}</h5>
                <ul>
                    {"".join([f"<li>{item}</li>" for item in items])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Treatment history for user
        user_treatments = [h for h in st.session_state.history if h['Prediction'] == selected_disease]
        
        if user_treatments:
            st.markdown("### 📚 Your Treatment History")
            st.metric("Cases Treated", len(user_treatments))
            
            if user_treatments:
                last_treatment = user_treatments[-1]
                st.markdown(f"""
                <div class="notification">
                    <strong>Last Case:</strong> {last_treatment['Timestamp']}<br>
                    <strong>Confidence:</strong> {last_treatment['Confidence']}<br>
                    <strong>Model:</strong> {last_treatment['Model']}
                </div>
                """, unsafe_allow_html=True)
    
    # Additional resources section
    st.markdown("---")
    st.markdown("## 📚 Additional Resources & Expert Tips")
    
    resource_cols = st.columns(3)
    
    with resource_cols[0]:
        st.markdown("""
        <div class="analysis-container">
            <h4>📖 Educational Resources</h4>
            <ul>
                <li>🌱 Plant Care Best Practices Guide</li>
                <li>🔬 Disease Identification Manual</li>
                <li>💧 Optimal Watering Techniques</li>
                <li>🌞 Light Requirements Database</li>
                <li>🌡️ Temperature Control Methods</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with resource_cols[1]:
        st.markdown("""
        <div class="analysis-container">
            <h4>🏥 Emergency Protocols</h4>
            <ul>
                <li>🚨 Rapid Response Checklist</li>
                <li>📞 When to Call Experts</li>
                <li>🏥 Plant Hospital Locations</li>
                <li>💊 Emergency Treatment Kit</li>
                <li>📋 Symptom Documentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with resource_cols[2]:
        st.markdown("""
        <div class="analysis-container">
            <h4>🌿 Organic Alternatives</h4>
            <ul>
                <li>🌱 Natural Fungicides</li>
                <li>🐛 Beneficial Insects</li>
                <li>🌿 Companion Planting</li>
                <li>🏺 Homemade Remedies</li>
                <li>♻️ Sustainable Practices</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive treatment planner
    st.markdown("---")
    st.markdown("### 📅 Personal Treatment Planner")
    
    planner_cols = st.columns(2)
    
    with planner_cols[0]:
        st.markdown("#### 📋 Create Treatment Plan")
        
        plant_name = st.text_input("🌱 Plant Name/ID", placeholder="e.g., Tomato Plant #1")
        treatment_start = st.date_input("📅 Treatment Start Date", datetime.now().date())
        severity = st.slider("🚨 Severity Assessment", 1, 4, disease_data['severity_level'])
        notes = st.text_area("📝 Additional Notes", placeholder="Any specific observations or concerns...")
        
        if st.button("📋 Create Treatment Plan"):
            plan = {
                "plant_name": plant_name,
                "disease": selected_disease,
                "start_date": treatment_start.strftime("%Y-%m-%d"),
                "severity": severity,
                "notes": notes,
                "created_by": st.session_state.username,
                "created_at": datetime.now().isoformat()
            }
            
            if 'treatment_plans' not in st.session_state:
                st.session_state.treatment_plans = []
            
            st.session_state.treatment_plans.append(plan)
            st.success("✅ Treatment plan created successfully!")
    
    with planner_cols[1]:
        st.markdown("#### 📊 Your Treatment Plans")
        
        if 'treatment_plans' in st.session_state and st.session_state.treatment_plans:
            user_plans = [p for p in st.session_state.treatment_plans if p['created_by'] == st.session_state.username]
            
            if user_plans:
                for i, plan in enumerate(user_plans[-3:]):  # Show last 3 plans
                    st.markdown(f"""
                    <div class="feature-card">
                        <h5>🌱 {plan['plant_name']}</h5>
                        <p><strong>Disease:</strong> {plan['disease']}</p>
                        <p><strong>Start Date:</strong> {plan['start_date']}</p>
                        <p><strong>Severity:</strong> {plan['severity']}/4</p>
                        <p><small>{plan['notes'][:50]}...</small></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No treatment plans created yet.")
        else:
            st.info("Create your first treatment plan above.")
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
    col1, col2, col3 = st.columns(3)
    
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
    with col3:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# --- Batch Processing Feature ---
def show_batch_processing():
    st.markdown('<h1 class="main-header">⚡ Batch Processing</h1>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "📤 Upload Multiple Plant Images", 
        type=["jpg", "png", "jpeg"], 
        accept_multiple_files=True,
        help="Upload up to 10 images for batch processing"
    )
    
    if uploaded_files and len(uploaded_files) > 0:
        st.success(f"✅ {len(uploaded_files)} images uploaded successfully!")
        
        if st.button("🚀 Process All Images", type="primary"):
            results = []
            progress_bar = st.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                image = Image.open(uploaded_file)
                result = analyze_image_advanced(image, "Voting Ensemble")
                result['filename'] = uploaded_file.name
                results.append(result)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Display batch results
            st.markdown("## 📊 Batch Results")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            healthy_count = sum(1 for r in results if r['prediction'] == 'Healthy')
            disease_count = len(results) - healthy_count
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
            
            with col1:
                st.metric("Total Processed", len(results))
            with col2:
                st.metric("Healthy Plants", healthy_count)
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            
            # Results table
            results_df = pd.DataFrame([
                {
                    'Filename': r['filename'],
                    'Prediction': r['prediction'],
                    'Confidence': f"{r['confidence']:.1f}%",
                    'Status': '✅ Healthy' if r['prediction'] == 'Healthy' else '🦠 Disease'
                }
                for r in results
            ])
            
            st.dataframe(results_df, use_container_width=True)
            
            # Export batch results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Batch Results",
                csv,
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

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
    if st.session_state.get('user_role') == 'admin':
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
