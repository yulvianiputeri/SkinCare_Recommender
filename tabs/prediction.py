"""Tab 3: Rating Prediction"""
import streamlit as st
import pandas as pd


def render_prediction(df, hybrid_system):
    """Render prediction tab"""
    st.header("🤖 Predict Product Rating")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
        return

    # Choose prediction mode
    pred_mode = st.radio(
        "Select Prediction Mode:",
        [" Predict Existing Product", " Predict New Product"],
        horizontal=True
    )
    
    if pred_mode == " Predict Existing Product":
        # MODE 1: EXISTING PRODUCT
        st.markdown("### Select a product to predict its rating")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            brand_filter = st.selectbox(
                "Filter by Brand (optional)",
                ["All Brands"] + sorted(df['brand_name'].unique()[:50])
            )
            
            if brand_filter != "All Brands":
                filtered_df = df[df['brand_name'] == brand_filter]
            else:
                filtered_df = df
            
            product_name = st.selectbox(
                "Select Product",
                filtered_df['product_name'].head(200)
            )
        
        with col2:
            if st.button(" Predict Rating", type="primary", use_container_width=True):
                try:
                    product = df[df['product_name'] == product_name].iloc[0]
                    product_idx = df[df['product_name'] == product_name].index[0]
                    original_idx = df.index.get_loc(product_idx)
                    
                    scores = hybrid_system.get_ensemble_score(original_idx)
                    predicted_rating = scores['ensemble_score'] * 5.0
                    actual_rating = product['average_rating']
                    
                    st.markdown("###  Prediction Results")
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Actual Rating", f"⭐ {actual_rating:.2f}/5.0")
                    with col_b:
                        diff = predicted_rating - actual_rating
                        st.metric("Predicted Rating", f"⭐ {predicted_rating:.2f}/5.0", delta=f"{diff:+.2f}")
                    
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
        # MODE 2: NEW PRODUCT
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
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                <h2>Predicted Rating</h2>
                <h1 style="font-size: 4rem; margin: 1rem 0;">⭐ {predicted_rating:.2f}/5.0</h1>
                </div>
                """, unsafe_allow_html=True)
                
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
                    st.success("🎉 **Excellent Potential!** This product is predicted to receive very high ratings.")
                elif predicted_rating >= 4.0:
                    st.info(" **Good Potential** Product quality expected to be well-received.")
                elif predicted_rating >= 3.5:
                    st.warning("⚠️ **Average Potential** May face moderate competition.")
                else:
                    st.error("❌ **Low Potential** Predicted rating is below average.")
                
                # Recommendations
                st.markdown("###  Recommendations")
                
                if reviews < 50:
                    st.info(" Consider strategies to boost initial reviews")
                if wishlist < 30:
                    st.info(" Build pre-launch hype to increase wishlist count")
                if predicted_rating < 4.0:
                    st.warning(" Review product formulation and compare with similar high-rated products")
            
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())