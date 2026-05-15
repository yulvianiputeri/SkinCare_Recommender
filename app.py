"""Skincare Recommender - Main Streamlit Application"""
import streamlit as st
import pandas as pd
import numpy as np
import os

# Import models
try:
    from ensemble_system import EnsembleHybridSystem as HybridRecommenderSystem
except ImportError:
    from hybrid_model import HybridRecommenderSystem

from analytics import SkincareAnalytics
from tfidf_model import TFIDFContentModel
from svd_model import SVDCollaborativeModel
from rf_model_improved import RandomForestModel

# Import tabs
from tabs import (
    render_recommendations,
    render_similar_products,
    render_prediction,
    render_comparison,
    render_analytics,
    render_preprocessing
)

# Page config
st.set_page_config(page_title="🧴 Skincare ML", page_icon="🧴", layout="wide")

# Load CSS from file
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    try:
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("styles.css not found")

load_css()

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None

@st.cache_data
def load_data():
    """Load processed data"""
    paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
    for path in paths:
        try:
            df = pd.read_csv(path)
            # Filter out Unknown brand from display (but keep in dataset for consistency)
            df = df[df['brand_name'] != 'Unknown']
            return df
        except FileNotFoundError:
            continue
    return None

@st.cache_resource
def get_hybrid_system():
    return HybridRecommenderSystem()

st.title("🧴 Skincare Recommender")
st.markdown("**TF-IDF + SVD Matrix Factorization + Random Forest Ensemble**")

# Load data
df = load_data()
if df is None:
    st.stop()

st.session_state.df = df
hybrid_system = get_hybrid_system()

# Sidebar
st.sidebar.header("Controls")
st.sidebar.metric("Products", f"{len(df):,}")
st.sidebar.metric("Brands", f"{df['brand_name'].nunique()}")
st.sidebar.metric("Categories", f"{df['default_category'].nunique()}")

dataset_type = "Full Dataset" if len(df) > 3000 else "Sample Dataset"
st.sidebar.info(f"📊 Using: **{dataset_type}**")

# Model controls
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Models")

model_files = ['models/tfidf_model.pkl', 'models/rf_model.pkl']
models_exist = all(os.path.exists(f) for f in model_files)

if not st.session_state.models_trained:
    if models_exist:
        if st.sidebar.button("📥 Load Models", use_container_width=True):
            with st.spinner("Loading..."):
                if hybrid_system.load_trained_models():
                    st.session_state.models_trained = True
                    st.rerun()
        
        st.sidebar.markdown("_atau_")
        
        if st.sidebar.button("⚠️ Retrain", use_container_width=True):
            with st.spinner("Training..."):
                hybrid_system.train_all_models()
                st.session_state.models_trained = True
                st.rerun()
    else:
        if st.sidebar.button("⚠️ Train Models", type="primary", use_container_width=True):
            with st.spinner("Training..."):
                hybrid_system.train_all_models()
                st.session_state.models_trained = True
                st.rerun()
else:
    st.sidebar.success("✅ Models Ready")

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><h3>📊 Products</h3><h2>{len(df):,}</h2></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><h3>🏢 Brands</h3><h2>{df["brand_name"].nunique()}</h2></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><h3>📋 Categories</h3><h2>{df["default_category"].nunique()}</h2></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><h3>⭐ Rating</h3><h2>{df["average_rating"].mean():.2f}</h2></div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎯 Recommendations", 
    "🔍 Similar Products", 
    "🤖 Prediction", 
    "📊 Model Comparison",  
    "📈 Analytics",
    "🔧 Preprocessing"
])

with tab1:
    render_recommendations(df, hybrid_system)

with tab2:
    render_similar_products(df, hybrid_system)

with tab3:
    render_prediction(df, hybrid_system)

with tab4:
    render_comparison(df, hybrid_system)

with tab5:
    render_analytics(df)

with tab6:
    render_preprocessing()