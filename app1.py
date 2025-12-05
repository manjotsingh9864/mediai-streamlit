from visualizations import render_visualizations
from auth import render_login
def draw_confidence_gauge(confidence, save_path):
    fig, ax = plt.subplots(figsize=(3,3), subplot_kw=dict(aspect="equal"))
    sizes = [float(confidence), max(0.0, 1.0 - float(confidence))]
    colors = ["#10B981", "#E5E7EB"]
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, wedgeprops=dict(width=0.4, edgecolor='white'))
    ax.text(0, 0, f"{confidence:.0%}", ha='center', va='center', fontsize=18, weight='bold')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", transparent=True)
    plt.close(fig)
import streamlit as st
import os
API_KEY = os.getenv("GEMINI_API_KEY")
import base64
TMP_DIR = os.path.join("temp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
from fpdf import FPDF
import textwrap
import tempfile
import json
from datetime import datetime
import joblib
import sqlite3
from about_us import render_about_us
DB_PATH = os.path.join("data", "history.db")
os.makedirs("data", exist_ok=True)

# Helper function to convert image to base64
def get_image_base64(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.warning(f"Image file '{image_path}' not found in the current directory.")
        return ""
    except Exception as e:
        st.warning(f"Error loading image: {e}")
        return ""

# SHAP for explainability (optional)
try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

# --- Google Gemini / GenAI initialization (robust) ---

GENAI_CLIENT = None
GENAI_MODEL = None
GENAI_OLD = None
try:
    from google import genai as genai_new
    client = genai_new.Client(api_key=API_KEY)
    try:
        for m in client.models.list():
            if "generateContent" in getattr(m, "supported_actions", []):
                GENAI_CLIENT = client
                GENAI_MODEL = m.name
                break
    except Exception:
        for cand in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]:
            try:
                _ = client.models.generate_content(model=cand, contents="Hello")
                GENAI_CLIENT = client
                GENAI_MODEL = cand
                break
            except Exception:
                continue
except Exception:
    GENAI_CLIENT = None
if GENAI_CLIENT is None:
    try:
        import google.generativeai as genai_old
        genai_old.configure(api_key=API_KEY)
        for cand in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]:
            try:
                _ = genai_old.GenerativeModel(cand)
                GENAI_OLD = genai_old
                GENAI_MODEL = cand
                break
            except Exception:
                continue
    except Exception:
        GENAI_OLD = None

def generate_gemini_reply(prompt: str, max_output_tokens: int = 512):
    if GENAI_CLIENT:
        resp = GENAI_CLIENT.models.generate_content(model=GENAI_MODEL, contents=prompt, max_output_tokens=max_output_tokens)
        return resp.text
    elif GENAI_OLD:
        mdl = GENAI_OLD.GenerativeModel(GENAI_MODEL)
        resp = mdl.generate_content(prompt)
        return resp.text
    else:
        raise RuntimeError("No Google GenAI client available. Install 'google-genai' or 'google-generativeai' and set GEMINI_API_KEY.")


# sklearn imports with graceful fallback
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

from utils.common import (
    load_data, load_model, build_symptom_dict, get_label_decoder,
    fetch_recommendations, topk_predictions
)

# Set Streamlit page config
st.set_page_config(page_title="AI Medical Recommendation System",
                   page_icon="🩺", layout="wide")

# --------- AUTH STATE DEFAULTS ---------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- AUTH GATE ----------------
if not st.session_state.logged_in:
    render_login()
    st.stop()
# Inject dark-mode professional CSS with blue-purple gradient buttons
st.markdown("""
<style>
body, .stApp { background-color: #0f172a; color: #f1f5f9; }
/* Remove white border/margin around main content */
.block-container {
    padding: 1rem 2rem;
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
}
.stApp {
    background-color: #0f172a !important;
}
.report-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 8px 12px;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.5);
    text-align: center;
    margin: 4px;
}
.report-card h4 { font-size: 0.8rem; color: #94a3b8; margin: 0; }
.report-card h2 { font-size: 1.2rem; color: #f8fafc; margin: 4px 0; }
.small-muted { color: #64748b; font-size: 0.7rem; }
/* Sidebar: keep light bluish gradient for contrast */
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #e0f2ff, #bae6fd, #7dd3fc);
    color: #0f172a !important;
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 15px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(180, 200, 255, 0.18);
}
[data-testid="stSidebar"] .sidebar-content {
    padding: 10px;
    color: #0f172a !important;
}
.stSelectbox, .stTextInput, .stNumberInput, .stTextArea {
    background-color: #334155 !important;
    color: #f1f5f9 !important;
    border: 1px solid #475569 !important;
    border-radius: 6px;
}
/* Dark blue to deep purple gradient for ALL buttons, including sidebar navigation and main page */
.stButton>button, .sidebar-nav-btn, button, input[type="button"], input[type="submit"] {
    background: linear-gradient(135deg, #1e3a8a 0%, #9333ea 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(30,58,138,0.16);
    transition: background 0.2s, box-shadow 0.2s;
    font-size: 1rem !important;
}
.stButton>button:hover, .sidebar-nav-btn:hover, button:hover, input[type="button"]:hover, input[type="submit"]:hover {
    background: linear-gradient(135deg, #2563eb 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.19);
}
</style>
""", unsafe_allow_html=True)

# Hide Streamlit default top header bar
st.markdown("""
<style>
/* Hide Streamlit default top header bar */
header[data-testid="stHeader"] {
    /* display: none; */
}
</style>
""", unsafe_allow_html=True)


# ---------------- Load Data & Model ----------------

description, precautions, medications, diets, workout, training, symptoms_df, severity = load_data()
base_model = load_model()
decode_label = get_label_decoder(base_model)
symptoms_dict = build_symptom_dict(training)
all_symptoms = list(symptoms_dict.keys())

# ------------- Build Weighted Vector (move up; used in Home & elsewhere) -------------
def build_weighted_vector(selected, severity_map):
    vec = np.zeros(len(symptoms_dict))
    for s in selected:
        idx = symptoms_dict.get(s)
        if idx is None:
            continue
        weight = severity_map.get(s, 1.0)
        vec[idx] = weight
    return vec



# ----------- Admin-only retrain section -----------
if st.session_state.get("role") == "admin" and SKLEARN_OK:
    st.sidebar.markdown("---")
    st.sidebar.subheader("👩‍🔬 Admin: Retrain Model")
    st.sidebar.caption("Upload CSV with same feature layout; last column is label.")
    up = st.sidebar.file_uploader("Training CSV", type=["csv"], key="admin_csv")
    if up is not None:
        import io
        df_new = pd.read_csv(up)
        if df_new.shape[1] >= 2:
            try:
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import LabelEncoder
                Xn = df_new.iloc[:, :-1].values
                yn_raw = df_new.iloc[:, -1].values
                le = LabelEncoder()
                yn = le.fit_transform(yn_raw)
                model_new = LogisticRegression(max_iter=600)
                model_new.fit(Xn, yn)
                # Save to disk and set as active
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", "custom_model.pkl")
                joblib.dump({"model": model_new, "classes_": le.classes_}, model_path)
                st.session_state["custom_model"] = model_new
                st.session_state["custom_decoder"] = lambda idx: le.inverse_transform([idx])[0]
                st.success("New model trained and activated for this session.")
            except Exception as e:
                st.error(f"Retraining failed: {e}")
        else:
            st.error("CSV must have at least 2 columns (features + label).")

# Sidebar navigation button style (override for sidebar-nav-btn, dark gradient)
st.sidebar.markdown("""
<style>
.sidebar-nav-btn {
    width: 98%;
    margin-bottom: 6px !important;
    background: linear-gradient(135deg, #1e3a8a 0%, #9333ea 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(30,58,138,0.16);
    transition: background 0.2s, box-shadow 0.2s;
    font-size: 1rem !important;
}
.sidebar-nav-btn:hover {
    background: linear-gradient(135deg, #2563eb 0%, #a855f7 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.19);
}
</style>
""", unsafe_allow_html=True)

# Updated sidebar menu container with visually enclosed buttons and heading
sidebar_nav_items = [
    {"label": "🏠 Dashboard", "tab_idx": 0, "key": "sidebar_nav_home"},
    {"label": "🔎 Symptom Checker", "tab_idx": 7, "key": "sidebar_nav_symptom"},
    {"label": "🧬 Explore Diseases", "tab_idx": 1, "key": "sidebar_nav_disease"},
    {"label": "📊 Visual Insights", "tab_idx": 2, "key": "sidebar_nav_viz"},
    {"label": "🤖 Ask AI", "tab_idx": 4, "key": "sidebar_nav_ai"},
    {"label": "🏥👨‍⚕️ Find Hospitals & Doctors", "tab_idx": 8, "key": "sidebar_nav_hospital_reco"},
    {"label": "📁 My Reports", "tab_idx": 3, "key": "sidebar_nav_reports"},
    {"label": "📜 All Users Data", "tab_idx": 5, "key": "sidebar_nav_allusers"},
    {"label": "ℹ️ About This App", "tab_idx": 6, "key": "sidebar_nav_about"},
]
from hospital_recommendation import render_hospital_recommendation
if "sidebar_selected_tab" not in st.session_state:
    st.session_state["sidebar_selected_tab"] = 0

## Redesigned Sidebar menu container (creative, vibrant blue gradient, inner shadow, glow, hover, etc.)
st.sidebar.markdown("""
<style>
.sidebar-health-menu-container {
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 55%, #7dd3fc 100%);
    border-radius: 18px;
    padding: 22px 16px 18px 16px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.18);
    margin-bottom: 20px;
    display: flex;
    flex-direction: column;
    align-items: stretch;
}
.sidebar-health-menu-heading {
    background: linear-gradient(135deg, #38bdf8, #0ea5e9, #3b82f6);
    color: #fff;
    font-weight: 900;
    font-size: 1.48rem;
    text-align: center;
    margin-bottom: 18px;
    letter-spacing: 0.02em;
    border-radius: 14px;
    padding: 12px 0;
    cursor: pointer;
    user-select: none;
    box-shadow: 0 2px 8px 0 rgba(59,130,246,0.18), 0 1.5px 0.5px 0 rgba(14,165,233,0.09);
    text-shadow: 0 2px 4px rgba(0,0,0,0.25);
    transition: all 0.25s cubic-bezier(0.4,0.2,0.2,1);
    position: relative;
    overflow: hidden;
}
.sidebar-health-menu-heading:hover {
    transform: scale(1.03);
    filter: brightness(1.06) drop-shadow(0 0 8px #38bdf8aa);
}
.sidebar-health-menu-heading span {
    font-size: 1.6em;
    margin-right: 0.15em;
    vertical-align: middle;
    filter: drop-shadow(0 1px 2px #0ea5e9aa);
}
.sidebar-health-menu-btns {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
</style>
<div class="sidebar-health-menu-container">
  <div class="sidebar-health-menu-heading">
    <span>🏥</span> Health Center Menu
  </div>
  <div class="sidebar-health-menu-btns">
""", unsafe_allow_html=True)

## Render all sidebar buttons inside this container, visually enclosed
for nav in sidebar_nav_items:
    if st.sidebar.button(
        nav["label"],
        key=nav["key"],
        help=f"Go to {nav['label']}",
        use_container_width=True,
    ):
        st.session_state["sidebar_selected_tab"] = nav["tab_idx"]

# Sidebar menu container close
st.sidebar.markdown("""
  </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info("**Disclaimer:** Educational use only — not medical advice.")

cache_decorator = getattr(st, "cache_resource", getattr(st, "cache", lambda f: f))
@cache_decorator
def train_helper_models(training_df):
    helpers = {}
    if training_df is None or training_df.empty or not SKLEARN_OK:
        return helpers
    X = training_df.iloc[:, :-1].values
    y = training_df.iloc[:, -1].values
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_train, y_train = X, y
        X_val, y_val = X, y
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rf.fit(X_train, y_train)
        try:
            rf_cal = CalibratedClassifierCV(rf, cv='prefit')
            rf_cal.fit(X_val, y_val)
            helpers['rf'] = rf_cal
        except Exception:
            helpers['rf'] = rf
    except Exception:
        pass
    try:
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_train, y_train)
        try:
            lr_cal = CalibratedClassifierCV(lr, cv='prefit')
            lr_cal.fit(X_val, y_val)
            helpers['lr'] = lr_cal
        except Exception:
            helpers['lr'] = lr
    except Exception:
        pass
    return helpers
helpers = train_helper_models(training)

## Remove tabs, use sidebar_selected_tab only.
# Ensure session state variable exists for sidebar navigation
if "sidebar_selected_tab" not in st.session_state:
    st.session_state["sidebar_selected_tab"] = 0

## ----------- MAIN PAGE RENDERING: use sidebar_selected_tab only -----------
if st.session_state["sidebar_selected_tab"] == 0:
    # Home page: Welcome, introduction, project overview, banner, features
    
    # Load images as base64
    manjot_image_base64 = get_image_base64("manjot.png")
    s_image_base64 = get_image_base64("s.png")
    
    st.markdown("""
    <style>
    .hero-section {
        position: relative;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 24px;
        padding: 0;
        margin-bottom: 40px;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(16,185,129,0.25), 0 0 100px rgba(59,130,246,0.15);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-background {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/png;base64,""" + s_image_base64 + """');
        background-size: cover;
        background-position: center;
        opacity: 0.15;
        filter: blur(2px);
        animation: subtleZoom 20s ease-in-out infinite alternate;
    }
    @keyframes subtleZoom {
        0% { transform: scale(1); }
        100% { transform: scale(1.1); }
    }
    .hero-overlay {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at 30% 50%, rgba(16,185,129,0.2) 0%, rgba(59,130,246,0.2) 50%, rgba(147,51,234,0.2) 100%);
        animation: colorShift 15s ease-in-out infinite;
    }
    @keyframes colorShift {
        0%, 100% { opacity: 0.7; }
        50% { opacity: 0.9; }
    }
    .hero-content {
        position: relative;
        z-index: 10;
        padding: 60px 40px;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .hero-image-container {
        position: relative;
        display: inline-block;
        margin-bottom: 30px;
        animation: heroFloat 4s ease-in-out infinite;
    }
    @keyframes heroFloat {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        25% { transform: translateY(-15px) rotate(2deg); }
        50% { transform: translateY(-8px) rotate(0deg); }
        75% { transform: translateY(-15px) rotate(-2deg); }
    }
    .hero-image-wrapper {
        position: relative;
        width: 250px;
        height: 250px;
        margin: 0 auto;
    }
    .hero-image {
        width: 100%;
        height: 100%;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid rgba(255,255,255,0.2);
        box-shadow: 
            0 0 40px rgba(16,185,129,0.6),
            0 0 80px rgba(59,130,246,0.4),
            0 0 120px rgba(147,51,234,0.3),
            inset 0 0 20px rgba(255,255,255,0.1);
        animation: imagePulse 3s ease-in-out infinite;
    }
    @keyframes imagePulse {
        0%, 100% { 
            box-shadow: 
                0 0 40px rgba(16,185,129,0.6),
                0 0 80px rgba(59,130,246,0.4),
                0 0 120px rgba(147,51,234,0.3),
                inset 0 0 20px rgba(255,255,255,0.1);
        }
        50% { 
            box-shadow: 
                0 0 60px rgba(16,185,129,0.8),
                0 0 100px rgba(59,130,246,0.6),
                0 0 140px rgba(147,51,234,0.5),
                inset 0 0 30px rgba(255,255,255,0.2);
        }
    }
    .hero-glow-ring {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 260px;
        height: 260px;
        border-radius: 50%;
        border: 3px solid transparent;
        background: linear-gradient(45deg, #10B981, #3B82F6, #9333EA, #10B981) border-box;
        -webkit-mask: linear-gradient(#fff 0 0) padding-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        animation: rotateGlow 8s linear infinite;
    }
    @keyframes rotateGlow {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    .hero-particles {
        position: absolute;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
    }
    .particle {
        position: absolute;
        width: 4px;
        height: 4px;
        background: radial-gradient(circle, rgba(16,185,129,0.8), transparent);
        border-radius: 50%;
        animation: float-particle 4s ease-in-out infinite;
    }
    .particle:nth-child(1) { top: 20%; left: 15%; animation-delay: 0s; }
    .particle:nth-child(2) { top: 40%; left: 85%; animation-delay: 1s; }
    .particle:nth-child(3) { top: 60%; left: 20%; animation-delay: 2s; }
    .particle:nth-child(4) { top: 80%; left: 75%; animation-delay: 1.5s; }
    @keyframes float-particle {
        0%, 100% { transform: translateY(0px) scale(1); opacity: 0.3; }
        50% { transform: translateY(-20px) scale(1.5); opacity: 0.8; }
    }
    .hero-title {
        font-size: 3.7rem;
        font-weight: 900;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 50%, #9333EA 100%);
        background-size: 220% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        animation: gradientFlow 5s ease infinite;
        text-shadow: 0 4px 20px rgba(59,130,246,0.25);
        letter-spacing: -1px;
        text-align: center;
    }
    @keyframes gradientFlow {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    .hero-subtitle {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 20px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        animation: fadeInUp 1s ease-out 0.3s both;
        text-align: center;
    }
    .hero-description {
        font-size: 1.25rem;
        line-height: 1.8;
        color: #e0e7ef;
        max-width: 800px;
        margin: 0 auto 30px;
        text-shadow: 0 1px 5px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out 0.6s both;
        text-align: center;
    }
    .hero-features-pills {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        align-items: center;
        gap: 18px;
        margin-top: 30px;
        animation: fadeInUp 1s ease-out 0.9s both;
    }
    .feature-pill {
        background: linear-gradient(90deg, #3B82F6 0%, #9333EA 100%);
        color: #fff;
        font-weight: 700;
        font-size: 1.1rem;
        padding: 14px 32px;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 18px rgba(59,130,246,0.15);
        transition: background 0.3s, box-shadow 0.3s, transform 0.18s;
        text-align: center;
        letter-spacing: 0.01em;
        cursor: pointer;
        filter: drop-shadow(0 0 8px #9333ea33);
        background-size: 200% auto;
        position: relative;
        z-index: 2;
    }
    .feature-pill:hover {
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 80%);
        box-shadow: 0 8px 30px rgba(16,185,129,0.25);
        transform: scale(1.06) translateY(-2px);
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .home-features-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 30px;
        margin-bottom: 40px;
        animation: fadeIn 1s ease-out 1.2s both;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .feature-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        border: 1px solid rgba(59,130,246,0.2);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        position: relative;
        overflow: hidden;
    }
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1));
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .feature-card:hover::before {
        opacity: 1;
    }
    .feature-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 50px rgba(16,185,129,0.2), 0 0 50px rgba(59,130,246,0.1);
        border-color: rgba(16,185,129,0.5);
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        display: inline-block;
        animation: iconBounce 2s ease-in-out infinite;
    }
    @keyframes iconBounce {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 12px;
        color: #f8fafc;
        position: relative;
        z-index: 1;
    }
    .feature-desc {
        font-size: 1rem;
        color: #cbd5e1;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    .home-section {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        border: 1px solid rgba(59,130,246,0.2);
    }
    .home-section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #10B981;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .home-section-title::before {
        content: '✨';
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section with Image and centered/gradient text and pills, plus creative footer
    st.markdown("""
    <style>
    .hero-footer {
        margin-top: 28px;
        text-align: center;
        font-size: 0.92rem;
        font-weight: 500;
        letter-spacing: 0.01em;
        background: linear-gradient(90deg, #a5b4fc 0%, #c7d2fe 30%, #f0abfc 60%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        filter: drop-shadow(0 1px 3px #9333ea33);
        padding: 0.7em 0 0.4em 0;
        border-radius: 10px;
        opacity: 0.9;
        transition: opacity 0.3s;
        user-select: none;
        font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    }
    .hero-footer:hover {
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-background"></div>
        <div class="hero-overlay"></div>
        <div class="hero-particles">
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
            <div class="particle"></div>
        </div>
        <div class="hero-content">
            <div class="hero-image-container">
                <div class="hero-image-wrapper">
                    <div class="hero-glow-ring"></div>
                    <img src="data:image/png;base64,{manjot_image_base64}" class="hero-image" alt="MediAI">
                </div>
            </div>
            <h1 class="hero-title">🩺 Welcome to MediAI</h1>
            <h2 class="hero-subtitle">Empowering Healthcare with AI-Driven Insights</h2>
            <p class="hero-description">
                Welcome to <strong>AI Medical Recommendation System</strong> – a cutting-edge platform that harnesses 
                the power of Artificial Intelligence to help you make informed, educational, and data-driven health decisions. 
                Experience the future of healthcare technology today.
            </p>
            <div class="hero-features-pills">
                <div class="feature-pill">🔎 Symptom Prediction</div>
                <div class="feature-pill">📚 Disease Explorer</div>
                <div class="feature-pill">📊 Visualizations</div>
                <div class="feature-pill">🤖 AI Assistant</div>
                <div class="feature-pill">📁 Reports &amp; History</div>
            </div>
            <div class="hero-footer">
                Created by Manjot Singh
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("""
    <div class="home-features-container">
        <div class="feature-card">
            <div class="feature-icon">🔎</div>
            <div class="feature-title">Symptom Input & Prediction</div>
            <div class="feature-desc">
                Input your symptoms and receive AI-powered disease predictions with confidence scores and detailed recommendations.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📚</div>
            <div class="feature-title">Disease Explorer</div>
            <div class="feature-desc">
                Browse a comprehensive database of diseases, view descriptions, precautions, medications, diets, and compare conditions side-by-side.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Visual Insights</div>
            <div class="feature-desc">
                Discover interactive charts, heatmaps, and networks that reveal patterns in symptoms, diseases, and predictions.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">💬</div>
            <div class="feature-title">AI Medical Assistant</div>
            <div class="feature-desc">
                Ask health-related questions and receive educational, AI-generated answers and tips in real-time.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">📁</div>
            <div class="feature-title">Reports & History</div>
            <div class="feature-desc">
                Download detailed reports of your session, review your prediction history, and export data for further analysis.
            </div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔐</div>
            <div class="feature-title">Secure & Private</div>
            <div class="feature-desc">
                Your data is protected with authentication and privacy by design. Only you (and admins) can access your history.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Why Choose Section
    st.markdown("""
    <div class="home-section">
        <div class="home-section-title">Why Choose Our System?</div>
        <ul style="font-size:1.1rem; color:#cbd5e1; line-height:2;">
            <li><b style="color:#10B981;">AI-driven accuracy:</b> Ensemble models and explainable AI for robust predictions.</li>
            <li><b style="color:#3B82F6;">Rich knowledge base:</b> Extensive disease and symptom database, curated recommendations.</li>
            <li><b style="color:#9333EA;">Interactive visualizations:</b> Heatmaps, networks, and charts for deeper insights.</li>
            <li><b style="color:#10B981;">Personalized reports:</b> Export your results in PDF/JSON for sharing or consultation.</li>
            <li><b style="color:#3B82F6;">Educational focus:</b> Designed for learning, awareness, and informed discussion – not a substitute for professional advice.</li>
            <li><b style="color:#9333EA;">Secure access:</b> Role-based authentication for privacy and admin features.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Get Started Section
    st.markdown("""
    <div class="home-section">
        <div class="home-section-title">Get Started</div>
        <ol style="font-size:1.1rem; color:#cbd5e1; line-height:2;">
            <li>Enter your symptoms and patient info using the sidebar.</li>
            <li>Navigate via the sidebar to explore features.</li>
            <li>Generate predictions, view recommendations, and interact with the AI Assistant.</li>
            <li>Download reports and review your session history anytime.</li>
        </ol>
        <p style="color:#fbbf24; background: rgba(251,191,36,0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #fbbf24; margin-top: 20px;">
            <strong>⚠️ Important Note:</strong> All outputs are for educational purposes only. Always consult a qualified medical professional for real health concerns.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("""
    <div style="text-align:center; margin-top:40px; padding: 30px; background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1)); border-radius: 16px; border: 2px solid rgba(16,185,129,0.3);">
        <h3 style="color:#10B981; font-size:1.8rem; font-weight:700; margin-bottom:15px;">
            Ready to Explore? 🚀
        </h3>
        <p style="color:#cbd5e1; font-size:1.2rem;">
            Use the <strong style="color:#3B82F6;">sidebar menu</strong> to begin your healthcare journey!
        </p>
    </div>
    """, unsafe_allow_html=True)


from disease_explorer import render_disease_explorer

if st.session_state["sidebar_selected_tab"] == 1:
    render_disease_explorer(description, precautions, medications, diets, workout)

if st.session_state["sidebar_selected_tab"] == 2:
    render_visualizations(training, st.session_state.get("history", []), DB_PATH)

## ----------- REPORTS SECTION -----------
if st.session_state["sidebar_selected_tab"] == 3:
    st.header("Reports & History")
    st.subheader("Session History")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df[["timestamp","disease","confidence"]])
        if st.button("Export all history as JSON"):
            out = json.dumps(st.session_state.history, indent=2).encode("utf-8")
            st.download_button("⬇️ Download history.json", data=out, file_name="history.json", mime="application/json")
    else:
        st.caption("No predictions in this session yet.")

## ----------- AI ASSISTANT SECTION -----------
if st.session_state["sidebar_selected_tab"] == 4:
    st.header("💬 AI Medical Assistant (Gemini)")
    st.write("Ask any question about diseases, symptoms, diet, medications, or health tips (educational only).")
    user_question = st.text_input("Type your question here:", key="ai_user_question_input_final")
    context_options = ["General medical education", "Disease symptoms", "Diet & lifestyle", "Medications (educational only)"]
    context_choice = st.selectbox("Select context (helps AI focus)", context_options, index=0, key="ai_context_choice_selectbox_secondary")
    if st.button("Get AI Answer", key="ai_assistant_button_final"):
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                ai_answer = generate_gemini_reply(user_question)
                st.markdown(f"**AI Answer:** {ai_answer}")
                st.markdown("**Tips:**")
                st.markdown("- Educational only, not a medical diagnosis.")
                st.markdown("- Consult a qualified professional for real health concerns.")
                if "ai_history" not in st.session_state:
                    st.session_state.ai_history = []
                st.session_state.ai_history.append({"question": user_question, "answer": ai_answer})
                if st.session_state.ai_history:
                    st.markdown("### Recent AI interactions")
                    for i, qa in enumerate(reversed(st.session_state.ai_history[-5:])):
                        st.markdown(f"**Q{i+1}:** {qa['question']}")
                        st.markdown(f"**A{i+1}:** {qa['answer']}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error fetching AI answer: {e}")

## ----------- ALL USERS HISTORY SECTION -----------
if st.session_state["sidebar_selected_tab"] == 5:
    st.header("📜 All Users History")
    st.info("All Users History feature coming soon or restricted to admins.")

## ----------- ABOUT US SECTION -----------
if st.session_state["sidebar_selected_tab"] == 6:
    render_about_us()

# Disease Explorer Tab (Tab 1)
if st.session_state["sidebar_selected_tab"] == 1:
    st.header("Disease Explorer")
    diseases_available = sorted(description["Disease"].dropna().unique()) if not description.empty else []
    disease_pick = st.selectbox("Choose disease to inspect", options=diseases_available, key="disease_pick_selectbox_secondary")
    if disease_pick:
        desc, pre, meds, diet, wrk = fetch_recommendations(disease_pick, description, precautions, medications, diets, workout)
        st.subheader(disease_pick)
        st.write(desc if desc else "No description.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🛡️ Precautions")
            if pre:
                for p in pre: st.write("-", p)
            else: st.caption("None")
        with col2:
            st.markdown("### 💊 Medications")
            if meds:
                for m in meds: st.write("•", m)
            else: st.caption("None")
        with col3:
            st.markdown("### 🥗 Diet")
            if diet:
                for d in diet: st.write("•", d)
            else: st.caption("None")
        st.markdown("### 🏃 Workouts")
        if wrk:
            for w in wrk: st.write("•", w)
        else: st.caption("None")

    # Comparison mode
    st.markdown("---")
    st.subheader("Compare two diseases")
    comp1 = st.selectbox("Disease A", options=diseases_available, key="disease_compare_a_secondary")
    comp2 = st.selectbox("Disease B", options=diseases_available, key="disease_compare_b_secondary")
    if st.button("Compare", key="compare_disease_button_secondary"):
        d1 = fetch_recommendations(comp1, description, precautions, medications, diets, workout)
        d2 = fetch_recommendations(comp2, description, precautions, medications, diets, workout)

        categories = ["Precautions", "Medications", "Diet", "Workouts"]
        items1 = [d1[1] or [], d1[2] or [], d1[3] or [], d1[4] or []]
        items2 = [d2[1] or [], d2[2] or [], d2[3] or [], d2[4] or []]

        # Side-by-side display
        colA, colB = st.columns(2)
        with colA:
            st.write(f"### {comp1}")
            st.write(d1[0] or "No description")
            for cat, itms in zip(categories, items1):
                st.markdown(f"**{cat}**")
                if itms:
                    for i in itms:
                        st.write("-", i)
        with colB:
            st.write(f"### {comp2}")
            st.write(d2[0] or "No description")
            for cat, itms in zip(categories, items2):
                st.markdown(f"**{cat}**")
                if itms:
                    for i in itms:
                        st.write("-", i)

        # ✅ PERFECT grouped bar chart for comparing two diseases
        counts1 = [len(itms) for itms in items1]
        counts2 = [len(itms) for itms in items2]

        categories = ["Precautions", "Medications", "Diet", "Workouts"]
        x = np.arange(len(categories))
        width = 0.36

        fig, ax = plt.subplots(figsize=(7.5, 4.8))

        bars1 = ax.bar(
            x - width/2, counts1, width,
            label=comp1, color="#10B981"
        )
        bars2 = ax.bar(
            x + width/2, counts2, width,
            label=comp2, color="#3B82F6"
        )

        # Axis & title
        ax.set_ylabel("Number of Recommendations", fontsize=11)
        ax.set_xlabel("Categories", fontsize=11)
        ax.set_title("Recommendation Richness Comparison", fontsize=14, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)

        # Grid (clean & professional)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Value labels on top of bars
        def add_values(bars):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        h + 0.05,
                        str(int(h)),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        weight="bold"
                    )

        add_values(bars1)
        add_values(bars2)

        # Legend
        ax.legend(frameon=False, fontsize=10)

        plt.tight_layout(pad=1.6)
        st.pyplot(fig)

        # Highlight common vs unique items per category
        for cat, it1, it2 in zip(categories, items1, items2):
            common = set(it1).intersection(it2)
            unique1 = set(it1) - common
            unique2 = set(it2) - common
            st.markdown(f"### {cat} Comparison")
            st.write("**Common:**", list(common) if common else "None")
            st.write(f"**Unique to {comp1}:**", list(unique1) if unique1 else "None")
            st.write(f"**Unique to {comp2}:**", list(unique2) if unique2 else "None")

# Footer
st.markdown("---")
st.write("Built with ❤️ — enhance further by adding authentication, deployment, and more datasets.")

def init_history_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    username TEXT,
                    patient_name TEXT,
                    patient_age INTEGER,
                    patient_gender TEXT,
                    prediction TEXT,
                    confidence REAL
                )""")
    conn.commit()
    conn.close()
init_history_db()

def save_history_to_db(user, patient_name, patient_age, patient_gender, prediction, confidence):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (timestamp, username, patient_name, patient_age, patient_gender, prediction, confidence) VALUES (?,?,?,?,?,?,?)",
              (datetime.now().isoformat(), user, patient_name, patient_age, patient_gender, prediction, confidence))
    conn.commit()
    conn.close()

if "history" in st.session_state and st.session_state.history:
    current_len = len(st.session_state.history)
    last_saved_len = st.session_state.get("saved_history_len", 0)
    if current_len > last_saved_len:
        last_entry = st.session_state.history[-1]
        save_history_to_db(
            st.session_state.get("user","anon"),
            (last_entry["patient"].get("name") if isinstance(last_entry["patient"], dict) else last_entry["patient"]),
            (last_entry["patient"].get("age") if isinstance(last_entry["patient"], dict) else None),
            (last_entry["patient"].get("gender") if isinstance(last_entry["patient"], dict) else None),
            last_entry.get("prediction"),
            float(last_entry.get("confidence", 0.0))
        )
        st.session_state["saved_history_len"] = current_len

# All Users History Tab (Tab 5)
if st.session_state["sidebar_selected_tab"] == 5:
    st.header("📜 All Users Prediction History")
    # Filters
    colf1, colf2, colf3 = st.columns(3)
    with colf1:
        user_f = st.text_input("Filter by username contains", key="all_users_filter_username")
    with colf2:
        disease_f = st.text_input("Filter by disease contains", key="all_users_filter_disease")
    with colf3:
        days = st.number_input("Last N days (0 = all)", min_value=0, max_value=3650, value=0, key="all_users_filter_days")
    # Build query
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT * FROM history"
    clauses = []
    params = []
    if user_f:
        clauses.append("username LIKE ?")
        params.append(f"%{user_f}%")
    if disease_f:
        clauses.append("prediction LIKE ?")
        params.append(f"%{disease_f}%")
    if days and days > 0:
        since = (datetime.now() - pd.to_timedelta(days, unit="D")).isoformat()
        clauses.append("timestamp >= ?")
        params.append(since)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC"
    try:
        df_db_hist = pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()
    if not df_db_hist.empty:
        st.dataframe(df_db_hist)
        csv_bytes = df_db_hist.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV of all history", data=csv_bytes, file_name="all_history.csv", mime="text/csv")
    else:
        st.caption("No predictions stored yet or filters removed all rows.")

## ----------- SYMPTOM INPUT SECTION -----------
from symptom_input import render_symptom_input

# Ensure user_id and sel_row are defined for symptom_input module
user_id = st.session_state.get("user_email", "guest")
sel_row = None

if st.session_state["sidebar_selected_tab"] == 7:
    render_symptom_input(all_symptoms, base_model, decode_label, training, helpers, TMP_DIR, user_id, sel_row)

# Hospital & Doctor Recommendation Tab (Tab 8)
if st.session_state["sidebar_selected_tab"] == 8:
    render_hospital_recommendation()

# ----------- OPTIONAL WELCOME HEADER (after login, e.g., near Dashboard or always after auth) -----------
st.markdown(f"""
<div style="
    background: linear-gradient(135deg,#0f172a,#1e293b);
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 25px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.5);
">
    <h2 style="margin:0;color:#10B981">
        👋 Welcome, {st.session_state.get("user_name","User")}
    </h2>
    <p style="margin-top:8px;opacity:0.85">
        You are successfully logged in to MediAI
    </p>
</div>
""", unsafe_allow_html=True)

# ----------- OPTIONAL LOGOUT BUTTON (sidebar, recommended place) -----------
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Logout"):
    st.session_state.clear()
    st.rerun()