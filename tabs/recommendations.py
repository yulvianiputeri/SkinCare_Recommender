"""Tab 1: Product Recommendations"""
import streamlit as st
import pandas as pd
import numpy as np


def render_recommendations(df, hybrid_system):
    """Render recommendations tab"""
    st.header("🎯 Product Recommendations")
    st.markdown("Pilih produk referensi untuk menemukan produk serupa dengan skor detail.")

    if not st.session_state.models_trained:
        st.warning("⚠️ Training model terlebih dahulu melalui sidebar!")
        return

    # SECTION: PILIH PRODUK 
    st.subheader("1️⃣ Pilih Produk Referensi")
    
    # product_type = st.selectbox(
    #     "🎯 Tipe Produk:",
    #     ["Semua", "Skincare", "Makeup"],
    #     key="type_select_rec"
    # )
    # 
    # # Kategori Skincare
    # skincare_cats = ['Face Serum', 'Face Cream', 'Face Wash', 'Sheet Mask', 'Toner', 
    #                'Moisturizer', 'Sunscreen', 'Eye Cream', 'Essence', 'Serum', 
    #                'Tissue Mask', 'Lip Balm', 'Lip Care', 'Body Lotion', 'Body Cream', 
    #                'Body Wash', 'Hand Cream', 'Foot Cream', 'Beauty Supplements',
    #                'Face Mist', 'Face Gel', 'Face Oil', 'Sleeping Mask', 'Clay Mask',
    #                'Wash Off Mask', 'Hand Wash', 'Body Scrub', 'Body Oil', 'Body Butter',
    #                'Hand Sanitizer', 'Face Palette', 'Body Mist', 'Hair Mask', 'Hair Serum',
    #                'Hair Mist', 'Body Sunscreen', 'Hand & Foot Cream', 'Face Brushes']
    # 
    # # Kategori Makeup
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brand_filter = st.selectbox(
            "Filter Brand (Opsional)",
            ["Semua Brand"] + sorted(df['brand_name'].unique()),
            key="brand_filter_rec"
        )
    
    # Apply brand filter first
    if brand_filter != "Semua Brand":
        filtered_df = df[df['brand_name'] == brand_filter]
    else:
        filtered_df = df.copy()
    
    with col2:
        category_filter = st.selectbox(
            "Filter Kategori (Opsional)",
            ["Semua Kategori"] + sorted(filtered_df['default_category'].unique()),
            key="category_filter_rec"
        )
    
    if category_filter != "Semua Kategori":
        filtered_df = filtered_df[filtered_df['default_category'] == category_filter]
    
    with col3:
        product_name = st.selectbox(
            "Pilih Produk",
            filtered_df['product_name'].head(200),
            key="product_select_rec"
        )

    # Show selected product info
    selected_product = df[df['product_name'] == product_name].iloc[0]
    with st.expander("📋 Detail Produk Dipilih"):
        st.markdown(f"""
        <div class="rec-card">
        <p><b>Nama:</b> {selected_product['product_name']}</p>
        <p><b>Brand:</b> {selected_product['brand_name']}</p>
        <p><b>Kategori:</b> {selected_product['default_category']}</p>
        <p><b>Rating:</b> ⭐ {selected_product['average_rating']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    # ===== FIND BUTTON =====
    st.markdown("---")
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        find_clicked = st.button("🔍 Cari Produk Serupa", type="primary", use_container_width=True)

    # ===== RESULTS =====
    if find_clicked:
        try:
            with st.spinner("Menganalisis produk serupa..."):
                product_idx = df[df['product_name'] == product_name].index[0]
                similar = hybrid_system.get_similar_products(product_idx, n_recommendations=5)

                if similar.empty:
                    st.warning("Tidak ada produk serupa ditemukan")
                    return

                st.success(f"✅ Ditemukan 5 produk serupa!")
                
                # Collect results
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

                # ===== QUICK VIEW TABLE =====
                st.markdown("---")
                st.subheader("2️⃣ Hasil Rekomendasi")
                
                # Prepare table data
                table_data = []
                for i, p in enumerate(results, 1):
                    table_data.append({
                        "#": i,
                        "Produk": p["product_name"][:35] + "..." if len(p["product_name"]) > 35 else p["product_name"],
                        "Brand": p["brand_name"],
                        "Rating": f"⭐{p['average_rating']:.2f}",
                        "TF-IDF": f"{p['tfidf_score']:.3f}",
                        "SVD": f"{p['svd_score']:.3f}",
                        "RF": f"{p['rf_score']:.3f}",
                        "Ensemble": f"**{p['ensemble_score']:.3f}**"
                    })
                
                # Add average row
                avg_tfidf = np.mean([p["tfidf_score"] for p in results])
                avg_svd = np.mean([p["svd_score"] for p in results])
                avg_rf = np.mean([p["rf_score"] for p in results])
                avg_ensemble = np.mean([p["ensemble_score"] for p in results])
                
                table_data.append({
                    "#": "",
                    "Produk": "📊 Rata-rata",
                    "Brand": "",
                    "Rating": "",
                    "TF-IDF": f"{avg_tfidf:.3f}",
                    "SVD": f"{avg_svd:.3f}",
                    "RF": f"{avg_rf:.3f}",
                    "Ensemble": f"**{avg_ensemble:.3f}**"
                })
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

                # ===== DETAILED VIEW =====
                st.markdown("---")
                st.subheader("3️⃣ Detail Produk")
                
                for i, p in enumerate(results, 1):
                    price = f"Rp{p['price_numeric']:,}" if p["price_numeric"] > 0 else "N/A"
                    
                    st.markdown(f"""
                    <div class="rec-card">
                        <h4>{i}. 🧴 {p['product_name']}</h4>
                        <p><b>Brand:</b> {p['brand_name']} | <b>Kategori:</b> {p['default_category']}</p>
                        <p><b>Rating:</b> ⭐ {p['average_rating']:.2f} | <b>Harga:</b> {price}</p>
                        <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #eee;">
                        <p><b>Skor Model:</b><br>
                        TF-IDF: <span class="score-badge">{p['tfidf_score']:.3f}</span> | 
                        SVD: <span class="score-badge">{p['svd_score']:.3f}</span> | 
                        RF: <span class="score-badge">{p['rf_score']:.3f}</span> | 
                        <b>Ensemble: <span class="score-badge score-high">{p['ensemble_score']:.3f}</span></b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")