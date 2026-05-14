"""Tab 2: Similar Products"""
import streamlit as st


def render_similar_products(df, hybrid_system):
    """Render similar products tab"""
    st.header("🔍 Find Similar Products")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Train models first!")
        return

    # product_type = st.selectbox(
    #     "🎯 Tipe Produk:",
    #     ["Semua", "Skincare", "Makeup"],
    #     key="type_select_similar"
    # )
    # 
    # skincare_cats = ['Face Serum', 'Face Cream', 'Face Wash', 'Sheet Mask', 'Toner', 
    #                'Moisturizer', 'Sunscreen', 'Eye Cream', 'Essence', 'Serum', 
    #                'Tissue Mask', 'Lip Balm', 'Lip Care', 'Body Lotion', 'Body Cream', 
    #                'Body Wash', 'Hand Cream', 'Foot Cream', 'Beauty Supplements',
    #                'Face Mist', 'Face Gel', 'Face Oil', 'Sleeping Mask', 'Clay Mask',
    #                'Wash Off Mask', 'Hand Wash', 'Body Scrub', 'Body Oil', 'Body Butter',
    #                'Hand Sanitizer', 'Face Palette', 'Body Mist', 'Hair Mask', 'Hair Serum',
    #                'Hair Mist', 'Body Sunscreen', 'Hand & Foot Cream', 'Face Brushes']
    # 
    # makeup_cats = ['Lipstick', 'Lip Tint', 'Lip Cream', 'Lip Gloss', 'Eyeliner', 'Mascara',
    #              'Eyeshadow', 'Foundation', 'Concealer', 'Blush', 'Bronzer', 'Highlighter',
    #              'Powder', 'Pressed Powder', 'Loose Powder', 'Primer', 'BB Cream', 'CC Cream',
    #              'Makeup Remover', 'Nail Polish', 'Nail Arts', 'False Eyelashes', 
    #              'Eyebrows', 'Eye Brushes', 'Makeup Palettes', 'Lip Scrub', 'Lip Mask',
    #              'Travel Bottles', 'Eyelash Serum']
    # 
    # if product_type == "Skincare":
    #     df_filtered = df[df['default_category'].isin(skincare_cats)]
    # elif product_type == "Makeup":
    #     df_filtered = df[df['default_category'].isin(makeup_cats)]
    # else:
    #     df_filtered = df.copy()
    
    # Filter brand (opsional)
    col1, col2 = st.columns(2)
    with col1:
        brand_filter = st.selectbox(
            "Filter Brand (Opsional)",
            ["Semua Brand"] + sorted(df['brand_name'].unique()),
            key="brand_filter_similar"
        )
    
    if brand_filter != "Semua Brand":
        filtered_df = df[df['brand_name'] == brand_filter]
    else:
        filtered_df = df
    
    with col2:
        product_name = st.selectbox(
            "Pilih Produk",
            filtered_df['product_name'].head(100),
            key="product_select_similar"
        )
    
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