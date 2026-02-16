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
    st.info("ðŸ”„ Running baseline comparison... This may take 1-2 minutes")
    
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
    status_text.text("âœ… Comparison complete!")
    
    return pd.DataFrame(results)

def create_comparison_charts(results_df):
    """Create interactive comparison charts using Plotly"""
    
    # 3-panel comparison chart
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('RÂ² Score', 'MAE (Lower is Better)', 'RMSE (Lower is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    # RÂ² chart
    fig.add_trace(
        go.Bar(
            x=results_df['model'],
            y=results_df['r2'],
            name='RÂ²',
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

st.title("🧴 Skincare ML Recommender")
st.markdown("**TF-IDF + SVD Matrix Factorization + Random Forest Ensemble**")

df = load_data()
if df is None:
    st.stop()

st.session_state.df = df
hybrid_system = get_hybrid_system()

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
    
    if st.sidebar.button("⚠️ Train Models"):
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Recommendations", 
    "🔍 Similar Products", 
    "🤖 Prediction", 
    "📊 Model Comparison",  
    "📈 Analytics"
])

# TAB 1: RECOMMENDATIONS
with tab1:
    st.header("🎯 Product Recommendations")
    st.markdown("**Step 1:** Select a reference product | **Step 2:** Get similar products with detailed scores")

    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first using sidebar!")
    else:
        st.subheader("Select Reference Product")

        # Category filter
        category_filter = st.selectbox(
            "Filter by Category (optional)",
            ["All Categories"] + sorted(df['default_category'].unique())
        )

        if category_filter != "All Categories":
            filtered_df = df[df['default_category'] == category_filter]
        else:
            filtered_df = df

        product_name = st.selectbox(
            "Choose Product to Find Similar Items",
            filtered_df['product_name'].head(200)
        )

        selected_product = df[df['product_name'] == product_name].iloc[0]
        st.info(
            f"**Selected:** {selected_product['product_name']} | "
            f"**Brand:** {selected_product['brand_name']} | "
            f"**Category:** {selected_product['default_category']}"
        )

        if st.button("🔍 Find Similar Products", type="primary", use_container_width=True):
            try:
                with st.spinner("Analyzing similar products..."):
                    product_idx = df[df['product_name'] == product_name].index[0]
                    similar = hybrid_system.get_similar_products(product_idx, n_recommendations=5)

                    if similar.empty:
                        st.warning("No similar products found")
                        st.stop()

                    st.success("✅ Found 5 similar products!")
                    st.subheader("📊 Similar Products with Model Scores")

                    # COLLECT RESULTS
                    results = []
                    for idx in similar.index:
                        original_idx = df.index.get_loc(idx)
                        scores = hybrid_system.get_ensemble_score(original_idx)

                        p = df.loc[idx]
                        results.append({
                            "product_name": p["product_name"],
                            "brand_name": p["brand_name"],
                            "default_category": p["default_category"],
                            "average_rating": p["average_rating"],
                            "price_numeric": p.get("price_numeric", 0),
                            "tfidf_score": scores["tfidf_score"],
                            "svd_score": scores["svd_score"],
                            "rf_score": scores["rf_score"],
                            "ensemble_score": scores["ensemble_score"]
                        })

                    # QUICK VIEW TABLE
                    st.markdown("### 📊 Quick View Table")

                    table_data = []
                    for i, p in enumerate(results, 1):
                        table_data.append({
                            "No": i,
                            "Product": p["product_name"][:40] + "..." if len(p["product_name"]) > 40 else p["product_name"],
                            "Brand": p["brand_name"],
                            "Rating": f"{p['average_rating']:.2f}",
                            "TF-IDF": f"{p['tfidf_score']:.3f}",
                            "SVD": f"{p['svd_score']:.3f}",
                            "RF": f"{p['rf_score']:.3f}",
                            "Ensemble": f"{p['ensemble_score']:.3f}"
                        })

                    avg_tfidf = np.mean([p["tfidf_score"] for p in results])
                    avg_svd = np.mean([p["svd_score"] for p in results])
                    avg_rf = np.mean([p["rf_score"] for p in results])
                    avg_ensemble = np.mean([p["ensemble_score"] for p in results])

                    table_data.append({
                        "No": "",
                        "Product": "Rata-rata",
                        "Brand": "",
                        "Rating": "",
                        "TF-IDF": f"{avg_tfidf:.3f}",
                        "SVD": f"{avg_svd:.3f}",
                        "RF": f"{avg_rf:.3f}",
                        "Ensemble": f"{avg_ensemble:.3f}"
                    })

                    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                    # DETAILED VIEW
                    st.markdown("### 📋 Detailed View")

                    for i, p in enumerate(results, 1):
                        price = f"Rp{p['price_numeric']:,}" if p["price_numeric"] > 0 else "N/A"

                        st.markdown(f"""
                        <div class="rec-card">
                            <h4>{i}. 🧴 {p['product_name']}</h4>
                            <p><b>Brand:</b> {p['brand_name']} | <b>Category:</b> {p['default_category']}</p>
                            <p><b>Rating:</b> ⭐ {p['average_rating']:.2f} | <b>Price:</b> {price}</p>
                            <p><b>📊 Model Scores:</b><br>
                            TF-IDF: {p['tfidf_score']:.3f} |
                            SVD: {p['svd_score']:.3f} |
                            RF: {p['rf_score']:.3f} |
                            <b>Ensemble: {p['ensemble_score']:.3f}</b>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())



# Tab 2: Similar Products (existing - simplified)
with tab2:
    st.header("🔍 Find Similar Products")
    
    if not st.session_state.models_trained:
        st.warning("⚠️¸ Train models first!")
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
                        <p><b>Brand:</b> {p['brand_name']} | <b>Rating:</b> {p['average_rating']:.2f}</p>
                        <p><b>Similarity:</b> {p.get('similarity_score', 0):.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# Tab 3: Rating Prediction (FIXED - with 2 modes)
with tab3:
    st.header("Predict Product Rating")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        # Choose prediction mode
        pred_mode = st.radio(
            "Select Prediction Mode:",
            [" Predict Existing Product", " Predict New Product"],
            horizontal=True
        )
        
        if pred_mode == " Predict Existing Product":
            # MODE 1: EXISTING PRODUCT (ENSEMBLE)
            st.markdown("### Select a product to predict its rating")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Brand filter (optional)
                brand_filter = st.selectbox(
                    "Filter by Brand (optional)",
                    ["All Brands"] + sorted(df['brand_name'].unique()[:50])
                )
                
                if brand_filter != "All Brands":
                    filtered_df = df[df['brand_name'] == brand_filter]
                else:
                    filtered_df = df
                
                # Product selection
                product_name = st.selectbox(
                    "Select Product",
                    filtered_df['product_name'].head(200)
                )
            
            with col2:
                if st.button(" Predict Rating", type="primary", use_container_width=True):
                    try:
                        # Get product details
                        product = df[df['product_name'] == product_name].iloc[0]
                        
                        # Get actual rating and prediction
                        product_idx = df[df['product_name'] == product_name].index[0]
                        original_idx = df.index.get_loc(product_idx)
                        
                        scores = hybrid_system.get_ensemble_score(original_idx)
                        predicted_rating = scores['ensemble_score'] * 5.0
                        actual_rating = product['average_rating']
                        
                        # Display results
                        st.markdown("###  Prediction Results")
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Actual Rating", f"⭐ {actual_rating:.2f}/5.0")
                        with col_b:
                            diff = predicted_rating - actual_rating
                            st.metric("Predicted Rating", f"⭐ {predicted_rating:.2f}/5.0", delta=f"{diff:+.2f}")
                        
                        # Product details
                        price = f"Rp{product.get('price_numeric', 0):,}" if product.get('price_numeric', 0) > 0 else "N/A"
                        
                        st.markdown(f"""
                        <div class="rec-card">
                        <h4> Product Details</h4>
                        <p><b>Name:</b> {product['product_name']}</p>
                        <p><b>Brand:</b> {product['brand_name']} | <b>Category:</b> {product['default_category']}</p>
                        <p><b>Rating:</b> ⭐ {actual_rating:.2f}/5.0 | <b>Price:</b> {price}</p>
                        <p><b>Reviews:</b> {product['total_reviews']:,} | <b>Wishlist:</b> {product['total_in_wishlist']:,}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model scores breakdown
                        st.markdown("### 🔍 Model Scores Breakdown")
                        score_df = pd.DataFrame({
                            'Model': ['TF-IDF', 'SVD', 'RF', 'Ensemble'],
                            'Score (0-1)': [
                                scores['tfidf_score'],
                                scores['svd_score'],
                                scores['rf_score'],
                                scores['ensemble_score']
                            ],
                            'Rating (1-5)': [
                                scores['tfidf_score'] * 5.0,
                                scores['svd_score'] * 5.0,
                                scores['rf_score'] * 5.0,
                                predicted_rating
                            ]
                        })
                        st.dataframe(score_df, use_container_width=True, hide_index=True)
                        
                        # Interpretation
                        error = abs(predicted_rating - actual_rating)
                        
                        if error <= 0.1:
                            st.success("🎯 Excellent prediction accuracy! Error ≤ 0.1")
                        elif error <= 0.3:
                            st.info("✅ Good prediction accuracy. Error ≤ 0.3")
                        elif error <= 0.5:
                            st.warning("⚠️ Moderate accuracy. Error ≤ 0.5")
                        else:
                            st.error("❌ Prediction error > 0.5")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        else:  
            # MODE 2: NEW PRODUCT (RF METADATA)
            st.markdown("### Enter new product details for rating prediction")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_product_name = st.text_input("Product Name", placeholder="e.g., Anessa Sunscreen")
                brand = st.selectbox("Brand", sorted(df['brand_name'].unique()[:50]))
                category = st.selectbox("Category", sorted(df['default_category'].unique()))
            
            with col2:
                reviews = st.number_input("Expected Reviews", 0, 10000, 100, help="Estimated number of reviews")
                wishlist = st.number_input("Expected Wishlist", 0, 5000, 25, help="Estimated wishlist count")
                price = st.number_input("Price (Rp)", 0, 5000000, 150000, step=10000)
            
            if st.button(" Predict Rating for New Product", type="primary", use_container_width=True):
                try:
                    predicted_rating = hybrid_system.predict_rating(brand, category, reviews, wishlist, price)
                    
                    st.markdown("###  Prediction Results")
                    
                    # Big rating display
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h2>Predicted Rating</h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0;">⭐ {predicted_rating:.2f}/5.0</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Product summary
                    st.markdown(f"""
                    <div class="rec-card" style="margin-top: 1rem;">
                    <h4> New Product Summary</h4>
                    <p><b>Name:</b> {new_product_name}</p>
                    <p><b>Brand:</b> {brand} | <b>Category:</b> {category}</p>
                    <p><b>Expected Reviews:</b> {reviews:,} | <b>Expected Wishlist:</b> {wishlist:,}</p>
                    <p><b>Price:</b> Rp{price:,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Market potential analysis
                    st.markdown("### Market Potential Analysis")
                    
                    if predicted_rating >= 4.5:
                        st.success("🎉 **Excellent Potential!** This product is predicted to receive very high ratings. Strong market reception expected.")
                    elif predicted_rating >= 4.0:
                        st.info(" **Good Potential** Product quality expected to be well-received. Competitive in the market.")
                    elif predicted_rating >= 3.5:
                        st.warning("⚠️ **Average Potential** May face moderate competition. Consider product improvements or marketing strategy.")
                    else:
                        st.error("❌ **Low Potential** Predicted rating is below average. Recommend reconsidering product formulation or positioning.")
                    
                    # Recommendations
                    st.markdown("###  Recommendations")
                    
                    if reviews < 50:
                        st.info(" Consider strategies to boost initial reviews (early adopter programs, influencer marketing)")
                    if wishlist < 30:
                        st.info(" Build pre-launch hype to increase wishlist count before release")
                    if predicted_rating < 4.0:
                        st.warning(" Review product formulation and compare with similar high-rated products in category")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
# Tab 4: MODEL COMPARISON (NEW!)
with tab4:
    st.header("📊 Model Performance Comparison")
    
    st.markdown("""
    This tab compares the performance of individual models vs the hybrid ensemble system.
    
    **Models compared:**
    - **TF-IDF Only**: Content-based filtering using text similarity
    - **RF Only**: Random Forest using metadata (7 features)
    - **Hybrid Ensemble**: Combination of all models with meta-learner
    """)
    
    # Try to load existing results first
    existing_results = load_comparison_results()
    
    if existing_results is not None:
        st.success("Loaded existing comparison results")
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
            <p><b>RÂ²:</b> {best_model['r2']:.3f} | <b>MAE:</b> {best_model['mae']:.3f} | <b>RMSE:</b> {best_model['rmse']:.3f}</p>
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
                    st.metric("Ensemble RÂ²", f"{ensemble_r2:.3f}", 
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
    st.header("📊 Dataset Analytics")
    
    analytics = SkincareAnalytics()
    analytics.df = df
    analytics.render_dashboard()

st.markdown("---")
st.markdown("🧴 **Skincare ML Recommender** | Hybrid Ensemble Learning System | 2026")