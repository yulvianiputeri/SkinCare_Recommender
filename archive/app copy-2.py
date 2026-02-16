# Enhanced Streamlit App - Skincare ML Recommender with Model Comparison
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import models
try:
    from ensemble_system import EnsembleHybridSystem as HybridRecommenderSystem
except ImportError:
    from hybrid_model import HybridRecommenderSystem

from analytics import SkincareAnalytics
from tfidf_model import TFIDFContentModel
from svd_model import SVDCollaborativeModel
from rf_model_improved import RandomForestModel

# Page config
st.set_page_config(page_title="🧴 Skincare ML", page_icon="🧴", layout="wide")

# Simple CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white; text-align: center;
}
.rec-card {
    background: #f8f9fa; padding: 1rem; border-radius: 8px; 
    border-left: 4px solid #667eea; margin: 0.5rem 0;
}
.comparison-card {
    background: #e8f4fd; padding: 1.5rem; border-radius: 10px;
    border: 2px solid #2196f3; margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

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
            return df
        except FileNotFoundError:
            continue
    return None

@st.cache_resource
def get_hybrid_system():
    return HybridRecommenderSystem()

@st.cache_data
def load_comparison_results():
    """Load baseline comparison results if available"""
    try:
        results = pd.read_csv('baseline_comparison_results.csv')
        return results
    except:
        return None

def check_models():
    """Check if models are trained"""
    model_files = [
        'models/tfidf_model.pkl',
        'models/rf_model.pkl'
    ]
    return all(os.path.exists(f) for f in model_files)

def run_baseline_comparison(test_size=200):
    """Run baseline comparison with smaller test set for speed"""
    st.info("🔄 Running baseline comparison... This may take 1-2 minutes")
    
    df = st.session_state.df
    hybrid_system = get_hybrid_system()
    
    # Prepare test set
    test_indices = np.random.choice(len(df), min(test_size, len(df)), replace=False)
    y_test = df.iloc[test_indices]['average_rating'].values
    
    results = []
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 1. TF-IDF Only
    status_text.text("Evaluating TF-IDF...")
    progress_bar.progress(25)
    tfidf = TFIDFContentModel()
    if tfidf.load_model():
        predictions = []
        for idx in test_indices:
            score = tfidf.get_score(idx)
            predictions.append(score * 5.0)
        
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({
            'model': 'TF-IDF Only',
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
    
    # 2. RF Only
    status_text.text("Evaluating Random Forest...")
    progress_bar.progress(50)
    rf = RandomForestModel()
    if rf.load_model():
        predictions = []
        for idx in test_indices:
            score = rf.get_score(idx)
            predictions.append(score * 5.0)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({
            'model': 'RF Only',
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
    
    # 3. Ensemble
    status_text.text("Evaluating Ensemble...")
    progress_bar.progress(75)
    if hybrid_system.load_trained_models():
        predictions = []
        for idx in test_indices:
            scores = hybrid_system.get_ensemble_score(idx)
            predictions.append(scores['ensemble_score'] * 5.0)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({
            'model': 'Hybrid Ensemble',
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        })
    
    progress_bar.progress(100)
    status_text.text("✅ Comparison complete!")
    
    return pd.DataFrame(results)

def create_comparison_charts(results_df):
    """Create interactive comparison charts using Plotly"""
    
    # 3-panel comparison chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score', 'MAE (Lower is Better)', 'RMSE (Lower is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    # R² chart
    fig.add_trace(
        go.Bar(
            x=results_df['model'],
            y=results_df['r2'],
            name='R²',
            marker_color=colors,
            text=results_df['r2'].round(3),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # MAE chart
    fig.add_trace(
        go.Bar(
            x=results_df['model'],
            y=results_df['mae'],
            name='MAE',
            marker_color=colors,
            text=results_df['mae'].round(3),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # RMSE chart
    fig.add_trace(
        go.Bar(
            x=results_df['model'],
            y=results_df['rmse'],
            name='RMSE',
            marker_color=colors,
            text=results_df['rmse'].round(3),
            textposition='outside',
            showlegend=False
        ),
        row=1, col=3
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Model Performance Comparison",
        title_font_size=20
    )
    
    fig.update_xaxes(tickangle=-45)
    
    return fig

def create_improvement_chart(results_df):
    """Create improvement percentage chart"""
    if 'Hybrid Ensemble' not in results_df['model'].values:
        return None
    
    ensemble_r2 = results_df[results_df['model'] == 'Hybrid Ensemble']['r2'].values[0]
    
    improvements = []
    for idx, row in results_df.iterrows():
        if row['model'] != 'Hybrid Ensemble' and row['r2'] > 0:
            improvement = ((ensemble_r2 - row['r2']) / row['r2']) * 100
            improvements.append({
                'model': f"vs {row['model']}",
                'improvement': improvement
            })
    
    if not improvements:
        return None
    
    imp_df = pd.DataFrame(improvements)
    
    fig = px.bar(
        imp_df,
        x='improvement',
        y='model',
        orientation='h',
        title='Hybrid Ensemble Improvement (%)',
        color='improvement',
        color_continuous_scale='RdYlGn',
        text='improvement'
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=300, showlegend=False)
    fig.update_xaxes(title='Improvement (%)')
    fig.update_yaxes(title='')
    
    return fig

# Header
st.title("🧴 Skincare ML Recommender System")
st.markdown("Hybrid Ensemble: TF-IDF + SVD + Random Forest")

# Load data
df = load_data()
if df is None:
    st.error("❌ No data found! Run: `python data_pipeline.py`")
    st.stop()

st.session_state.df = df
hybrid_system = get_hybrid_system()

# Main app
st.title("🧴 Skincare ML Recommender")
st.markdown("**TF-IDF + SVD Matrix Factorization + Random Forest Ensemble**")

# Load data
df = load_data()
if df is None:
    st.stop()

st.session_state.df = df
hybrid_system = get_hybrid_system()

# Sidebar
st.sidebar.header("🎛️ Controls")
st.sidebar.metric("Products", f"{len(df):,}")
st.sidebar.metric("Brands", f"{df['brand_name'].nunique()}")
st.sidebar.metric("Categories", f"{df['default_category'].nunique()}")

dataset_type = "Full Dataset" if len(df) > 3000 else "Sample Dataset"
st.sidebar.info(f"📊 Using: **{dataset_type}**")

# Model controls
if not st.session_state.models_trained:
    if check_models():
        if st.sidebar.button("📥 Load Models"):
            with st.spinner("Loading models..."):
                if hybrid_system.load_trained_models():
                    st.session_state.models_trained = True
                    st.sidebar.success("✅ Models Loaded!")
                    st.rerun()
    
    if st.sidebar.button("🚀 Train Models"):
        with st.spinner("Training models... (2-3 minutes)"):
            try:
                metrics = hybrid_system.train_all_models()
                st.session_state.models_trained = True
                st.sidebar.success("✅ Training Complete!")
                st.sidebar.json(metrics)
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
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

# Tabs - ADDED MODEL COMPARISON TAB
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommendations", 
    "🔍 Similar Products", 
    "🤖 Prediction", 
    "📊 Model Comparison",  # NEW TAB
    "📈 Analytics"
])

# Tab 1: Recommendations (existing)
with tab1:
    st.header("🎯 Hybrid Recommendations")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first using sidebar!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Preferences")
            brands = st.multiselect("Brands", sorted(df['brand_name'].unique()[:50]))
            categories = st.multiselect("Categories", sorted(df['default_category'].unique()))
            min_rating = st.slider("Min Rating", 1.0, 5.0, 4.0, 0.1)
            
            prefs = {'brands': brands, 'categories': categories, 'min_rating': min_rating}
        
        with col2:
            if st.button("🎯 Get Recommendations", type="primary"):
                try:
                    with st.spinner("Generating recommendations..."):
                        recs = hybrid_system.get_recommendations(user_preferences=prefs, n_recommendations=5)
                    
                    if not recs.empty:
                        st.subheader("🏆 Your Recommendations")
                        
                        for i, (_, p) in enumerate(recs.iterrows(), 1):
                            price = f"Rp{p.get('price_numeric', 0):,}" if p.get('price_numeric', 0) > 0 else "N/A"
                            
                            st.markdown(f"""
                            <div class="rec-card">
                            <h4>{i}. 🧴 {p['product_name']}</h4>
                            <p><b>Brand:</b> {p['brand_name']} | <b>Category:</b> {p['default_category']}</p>
                            <p><b>Rating:</b> ⭐ {p['average_rating']:.2f} | <b>Price:</b> {price}</p>
                            <p><b>Ensemble Score:</b> {p.get('ensemble_score', 0):.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No recommendations found with these preferences")
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 2: Similar Products (existing - simplified)
with tab2:
    st.header("🔍 Find Similar Products")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        product_name = st.selectbox("Select Product", df['product_name'].head(100))
        
        if st.button("🔍 Find Similar"):
            try:
                product_idx = df[df['product_name'] == product_name].index[0]
                similar = hybrid_system.get_similar_products(product_idx, n_recommendations=5)
                
                if not similar.empty:
                    st.subheader("Similar Products")
                    for i, (_, p) in enumerate(similar.iterrows(), 1):
                        st.markdown(f"""
                        <div class="rec-card">
                        <h4>{i}. {p['product_name']}</h4>
                        <p><b>Brand:</b> {p['brand_name']} | <b>Rating:</b> ⭐ {p['average_rating']:.2f}</p>
                        <p><b>Similarity:</b> {p.get('similarity_score', 0):.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 3: Rating Prediction (existing - simplified)
with tab3:
    st.header("🤖 Predict Product Rating")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", sorted(df['brand_name'].unique()[:30]))
            category = st.selectbox("Category", sorted(df['default_category'].unique()))
        
        with col2:
            reviews = st.number_input("Expected Reviews", 0, 1000, 100)
            wishlist = st.number_input("Expected Wishlist", 0, 500, 50)
        
        if st.button("🤖 Predict Rating"):
            try:
                predicted_rating = hybrid_system.predict_rating(brand, category, reviews, wishlist)
                
                st.success(f"### Predicted Rating: ⭐ {predicted_rating:.2f}/5.0")
                
                if predicted_rating >= 4.5:
                    st.info("🎉 Excellent! High market potential")
                elif predicted_rating >= 4.0:
                    st.info("✅ Good quality expected")
                else:
                    st.warning("⚠️ May need improvement")
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 4: MODEL COMPARISON (NEW!)
with tab4:
    st.header("📊 Model Performance Comparison")
    
    st.markdown("""
    This tab compares the performance of individual models vs the hybrid ensemble system.
    
    **Models compared:**
    - 🔤 **TF-IDF Only**: Content-based filtering using text similarity
    - 🌳 **RF Only**: Random Forest using metadata (7 features)
    - 🔀 **Hybrid Ensemble**: Combination of all models with meta-learner
    """)
    
    # Try to load existing results first
    existing_results = load_comparison_results()
    
    if existing_results is not None:
        st.success("✅ Loaded existing comparison results")
        st.session_state.comparison_results = existing_results
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Options")
        test_size = st.slider("Test Size", 100, 500, 200, 50, 
                             help="Number of products to test (larger = more accurate but slower)")
        
        if st.button("🔄 Run Comparison", type="primary"):
            if not st.session_state.models_trained:
                st.error("⚠️ Train models first!")
            else:
                results = run_baseline_comparison(test_size)
                st.session_state.comparison_results = results
                # Save results
                results.to_csv('baseline_comparison_results.csv', index=False)
                st.success("✅ Results saved to baseline_comparison_results.csv")
                st.rerun()
    
    with col2:
        if st.session_state.comparison_results is not None:
            results_df = st.session_state.comparison_results
            
            # Display results table
            st.subheader("📋 Results Table")
            
            # Format the dataframe for display
            display_df = results_df.copy()
            display_df['r2'] = display_df['r2'].apply(lambda x: f"{x:.3f}")
            display_df['mae'] = display_df['mae'].apply(lambda x: f"{x:.3f}")
            display_df['rmse'] = display_df['rmse'].apply(lambda x: f"{x:.3f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Best model
            results_df_numeric = st.session_state.comparison_results
            best_idx = results_df_numeric['r2'].idxmax()
            best_model = results_df_numeric.iloc[best_idx]
            
            st.markdown(f"""
            <div class="comparison-card">
            <h3>🏆 Best Model: {best_model['model']}</h3>
            <p><b>R²:</b> {best_model['r2']:.3f} | <b>MAE:</b> {best_model['mae']:.3f} | <b>RMSE:</b> {best_model['rmse']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive charts
            st.subheader("📊 Interactive Charts")
            
            # Main comparison chart
            fig1 = create_comparison_charts(results_df_numeric)
            st.plotly_chart(fig1, use_container_width=True)
            
            # Improvement chart
            fig2 = create_improvement_chart(results_df_numeric)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            
            # Key insights
            st.subheader("💡 Key Insights")
            
            if 'Hybrid Ensemble' in results_df_numeric['model'].values:
                ensemble_r2 = results_df_numeric[results_df_numeric['model'] == 'Hybrid Ensemble']['r2'].values[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Ensemble R²", f"{ensemble_r2:.3f}", 
                             help="Higher is better (max 1.0)")
                
                with col2:
                    tfidf_r2 = results_df_numeric[results_df_numeric['model'] == 'TF-IDF Only']['r2'].values[0] if 'TF-IDF Only' in results_df_numeric['model'].values else 0
                    improvement = ((ensemble_r2 - tfidf_r2) / tfidf_r2 * 100) if tfidf_r2 > 0 else 0
                    st.metric("vs TF-IDF", f"{improvement:+.1f}%")
                
                with col3:
                    rf_r2 = results_df_numeric[results_df_numeric['model'] == 'RF Only']['r2'].values[0] if 'RF Only' in results_df_numeric['model'].values else 0
                    improvement = ((ensemble_r2 - rf_r2) / rf_r2 * 100) if rf_r2 > 0 else 0
                    st.metric("vs RF", f"{improvement:+.1f}%")
        else:
            st.info("👆 Click 'Run Comparison' to evaluate models")

# Tab 5: Analytics (existing)
with tab5:
    st.header("📈 Dataset Analytics")
    
    analytics = SkincareAnalytics()
    analytics.df = df
    analytics.render_dashboard()

# Footer
st.markdown("---")
st.markdown("🧴 **Skincare ML Recommender** | Hybrid Ensemble Learning System | 2026")