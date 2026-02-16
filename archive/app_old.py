# Simplified Streamlit App - Skincare ML Recommender
import streamlit as st
import pandas as pd
import numpy as np
import os
from hybrid_model import HybridRecommenderSystem
from analytics import SkincareAnalytics

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None

@st.cache_data
def load_data():
    """Load processed data"""
    paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
    for path in paths:
        try:
            df = pd.read_csv(path)
            st.success(f"✅ Loaded: {len(df):,} products from {path}")
            return df
        except FileNotFoundError:
            continue
    st.error("❌ No data found! Run: `python data_pipeline.py`")
    return None

@st.cache_resource
def get_hybrid_system():
    return HybridRecommenderSystem()

def check_models():
    """Check if models exist"""
    model_files = ['models/tfidf_model.pkl', 'models/svd_model.pkl', 'models/rf_model.pkl']
    return any(os.path.exists(f) for f in model_files)

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

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Similar Products", "🤖 Prediction", "📊 Analytics"])

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
                            
                            # Model scores
                            with st.expander(f"🤖 Model Scores", expanded=False):
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("TF-IDF", f"{p.get('tfidf_score', 0):.3f}")
                                with col2:
                                    st.metric("SVD", f"{p.get('svd_score', 0):.3f}")
                                with col3:
                                    st.metric("RF", f"{p.get('rf_score', 0):.3f}")
                                with col4:
                                    st.metric("Ensemble", f"{p.get('ensemble_score', 0):.3f}")
                    else:
                        st.warning("No recommendations found")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab2:
    st.header("🔍 Similar Products")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Product Selection")
            
            # Search by name
            search = st.text_input("🔍 Search Product", placeholder="Enter product name...")
            
            # Filter options
            filter_brand = st.selectbox("Filter by Brand", ["All"] + sorted(df['brand_name'].unique()[:50]))
            filter_category = st.selectbox("Filter by Category", ["All"] + sorted(df['default_category'].unique()))
            
            # Apply filters
            filtered_df = df.copy()
            
            if search:
                filtered_df = filtered_df[filtered_df['product_name'].str.contains(search, case=False, na=False)]
            
            if filter_brand != "All":
                filtered_df = filtered_df[filtered_df['brand_name'] == filter_brand]
                
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df['default_category'] == filter_category]
            
            if len(filtered_df) == 0:
                st.warning("No products found with current filters")
                selected_product_data = None
            else:
                # Product selection dropdown
                product_options = []
                for idx, row in filtered_df.head(20).iterrows():
                    product_options.append(f"{row['product_name'][:40]}... - {row['brand_name']}")
                
                if product_options:
                    selected_option = st.selectbox("Choose Product", product_options)
                    
                    if selected_option:
                        # Get selected product index
                        selected_idx = filtered_df.head(20).iloc[product_options.index(selected_option)].name
                        selected_product_data = df.loc[selected_idx]
                        
                        # Show selected product info
                        st.markdown("**📱 Selected Product:**")
                        st.info(f"**{selected_product_data['product_name'][:50]}{'...' if len(selected_product_data['product_name']) > 50 else ''}**")
                        st.markdown(f"**Brand:** {selected_product_data['brand_name']}")
                        st.markdown(f"**Category:** {selected_product_data['default_category']}")
                        st.markdown(f"**Rating:** ⭐ {selected_product_data['average_rating']:.2f}")
                        st.markdown(f"**Reviews:** 💬 {selected_product_data['total_reviews']:,}")
                        
                        # Show product position info
                        st.caption(f"Product #{df.index.get_loc(selected_idx) + 1} of {len(df):,}")
                    else:
                        selected_product_data = None
                else:
                    selected_product_data = None
        
        with col2:
            if st.button("🔍 Find Similar Products", type="primary") and 'selected_product_data' in locals() and selected_product_data is not None:
                with st.spinner("Finding similar products using TF-IDF..."):
                    try:
                        # Get product position in dataframe
                        product_position = df.index.get_loc(selected_idx)
                        similar_products = hybrid_system.get_similar_products(product_position, n_recommendations=6)
                        
                        if not similar_products.empty:
                            st.subheader("🔍 Similar Products Found")
                            st.success(f"Found {len(similar_products)} similar products using **TF-IDF Content Similarity**")
                            
                            for i, (_, s) in enumerate(similar_products.iterrows(), 1):
                                # Calculate why it's similar
                                similarity_reasons = []
                                if s['default_category'] == selected_product_data['default_category']:
                                    similarity_reasons.append("Same category")
                                if s['brand_name'] == selected_product_data['brand_name']:
                                    similarity_reasons.append("Same brand")
                                if abs(s['average_rating'] - selected_product_data['average_rating']) <= 0.5:
                                    similarity_reasons.append("Similar rating")
                                
                                reason_text = " + ".join(similarity_reasons) if similarity_reasons else "Content similarity"
                                
                                st.markdown(f"""
                                <div style="background: #e8f4fd; padding: 1rem; border-radius: 8px; border-left: 4px solid #2196f3; margin: 0.5rem 0;">
                                <h4>{i}. 🧴 {s['product_name'][:60]}{'...' if len(s['product_name']) > 60 else ''}</h4>
                                <p><b>Brand:</b> {s['brand_name']} | <b>Category:</b> {s['default_category']}</p>
                                <p><b>Rating:</b> ⭐ {s['average_rating']:.2f} | <b>Reviews:</b> 💬 {s['total_reviews']:,}</p>
                                <p><b>TF-IDF Similarity:</b> {s.get('similarity_score', 0):.3f} | <b>Why Similar:</b> {reason_text}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Show similarity analysis
                            avg_similarity = similar_products.get('similarity_score', [0]).mean()
                            unique_brands = similar_products['brand_name'].nunique()
                            unique_categories = similar_products['default_category'].nunique()
                            
                            st.markdown("---")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Similarity", f"{avg_similarity:.3f}")
                            with col2:
                                st.metric("Brands Found", f"{unique_brands}")
                            with col3:
                                st.metric("Categories", f"{unique_categories}")
                                
                            # Recommendation note
                            if avg_similarity >= 0.7:
                                st.success("🌟 High similarity! These products are excellent alternatives.")
                            elif avg_similarity >= 0.5:
                                st.info("👍 Good similarity! These products share common characteristics.")
                            else:
                                st.warning("⚠️ Moderate similarity. Consider browsing more options.")
                                
                        else:
                            st.warning("❌ No similar products found")
                            st.info("Try selecting a different product or adjusting your filters")
                            
                    except Exception as e:
                        st.error(f"❌ Error finding similar products: {e}")
                        st.info("This might be due to product indexing. Try selecting a different product.")
            
            elif 'selected_product_data' not in locals() or selected_product_data is None:
                st.info("👆 Please select a product from the left panel first")
            
            # Quick popular products shortcut
            if not ('selected_product_data' in locals() and selected_product_data is not None):
                st.markdown("---")
                st.subheader("🔥 Popular Products")
                st.markdown("*Click to explore similar products*")
                
                popular_products = df.nlargest(5, 'total_reviews')[['product_name', 'brand_name', 'average_rating', 'total_reviews']]
                
                for i, (idx, prod) in enumerate(popular_products.iterrows(), 1):
                    if st.button(f"{i}. {prod['product_name'][:40]}... - {prod['brand_name']}", key=f"pop_{i}"):
                        st.rerun()

with tab3:
    st.header("🤖 Rating Prediction")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Product Info")
            brand = st.selectbox("Brand", sorted(df['brand_name'].unique()[:30]))
            category = st.selectbox("Category", sorted(df['default_category'].unique()))
            reviews = st.number_input("Expected Reviews", 1, 1000, 50)
            wishlist = st.number_input("Expected Wishlist", 1, 500, 25)
        
        with col2:
            st.subheader("Prediction")
            if st.button("🔮 Predict Rating", type="primary"):
                try:
                    pred = hybrid_system.predict_rating(brand, category, reviews, wishlist)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                    <h2>🎯 Predicted Rating</h2>
                    <h1>{pred:.2f}/5.0</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if pred >= 4.5:
                        st.success("🌟 Exceptional potential!")
                    elif pred >= 4.0:
                        st.info("👍 Good potential")
                    else:
                        st.warning("⚠️ Average potential")
                except Exception as e:
                    st.error(f"Error: {e}")

with tab4:
    st.header("📊 Analytics")
    try:
        analytics = SkincareAnalytics()
        analytics.df = df
        analytics.render_dashboard()
    except Exception as e:
        st.error(f"Analytics error: {e}")

# Footer
st.markdown("---")
st.markdown(f"**🎯 Hybrid System** | {len(df):,} products | TF-IDF + SVD + RF Ensemble")