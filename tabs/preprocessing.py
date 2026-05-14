"""Preprocessing Steps Visualization Tab"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing_data import SkincareDataCleaner
from feature_engineering import SkincareFeatureEngineer


def render_preprocessing():
    """Render preprocessing steps tab"""
    st.header("🔧 Tahap Preprocessing Data")
    
    st.markdown("""
    Berikut adalah tahapan preprocessing data dari dataset Sociolla:
    """)
    
    # Step selector
    step = st.selectbox(
        "Pilih Tahap:",
        [
            "1. Load Data",
            "2. Drop Missing Values",
            "3. Clean Brand Names",
            "4. Clean Category Names",
            "5. Extract Price",
            "6. Filter Rating",
            "7. Remove Outliers",
            "8. Filter Rare Items",
            "9. Feature Engineering"
        ]
    )
    
    # Load data for demonstration
    @st.cache_data
    def load_raw_data():
        """Load raw dataset"""
        possible_paths = [
            'dataset/sociolla.csv',
            'sociolla.csv',
            'dataset/skincare_products.csv'
        ]
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                if len(df) > 100:
                    return df, path
            except:
                continue
        return None, None
    
    df_raw, path = load_raw_data()
    
    if df_raw is None:
        st.error("❌ Dataset tidak ditemukan!")
        return
    
    st.success(f"✅ Dataset loaded: {path} ({len(df_raw):,} rows)")
    
    # Show step details
    if step == "1. Load Data":
        st.subheader("📥 Tahap 1: Load Data")
        st.markdown("""
        **Deskripsi:** Memuat data asli dari file CSV
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
            st.metric("Total Columns", len(df_raw.columns))
        with col2:
            st.metric("Missing Values", df_raw.isna().sum().sum())
        
        st.subheader("Sample Data:")
        st.dataframe(df_raw.head(10), use_container_width=True)
        
        st.subheader("Columns:")
        for col in df_raw.columns:
            st.code(col)
    
    elif step == "2. Drop Missing Values":
        st.subheader("🗑️ Tahap 2: Drop Missing Values")
        st.markdown("""
        **Deskripsi:** Menghapus baris dengan missing values pada kolom essential:
        - `product_name`
        - `brand_name`  
        - `average_rating`
        """)
        
        essential_cols = ['product_name', 'brand_name', 'average_rating']
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
        with col2:
            st.metric("Missing product_name", df_raw['product_name'].isna().sum())
        with col3:
            st.metric("Missing brand_name", df_raw['brand_name'].isna().sum())
        
        st.subheader("Data BEFORE:")
        st.dataframe(df_raw[df_raw['product_name'].isna() | df_raw['brand_name'].isna() | df_raw['average_rating'].isna()].head(10), use_container_width=True)
        
        # AFTER
        df_dropped = df_raw.dropna(subset=essential_cols)
        
        st.subheader("📊 AFTER:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_dropped):,}")
        with col2:
            st.metric("Rows Removed", f"{len(df_raw) - len(df_dropped):,}")
        
        st.subheader("Data AFTER:")
        st.dataframe(df_dropped.head(10), use_container_width=True)
    
    elif step == "3. Clean Brand Names":
        st.subheader("🧼 Tahap 3: Clean Brand Names")
        st.markdown("""
        **Deskripsi:** Membersihkan nama brand:
        1. Hapus prefix numerik (contoh: `123_SomeBrand` → `SomeBrand`)
        2. Ambil bagian yang meaningful
        3. Capitalize huruf pertama
        """)
        
        cleaner = SkincareDataCleaner()
        
        df_clean = df_raw.dropna(subset=['brand_name'])
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        st.metric("Unique Brands", df_clean['brand_name'].nunique())
        
        st.subheader("Data BEFORE (Original Brand Names):")
        st.dataframe(df_clean[['product_name', 'brand_name']].drop_duplicates('brand_name').head(15), use_container_width=True)
        
        # AFTER
        df_clean['brand_cleaned'] = df_clean['brand_name'].apply(cleaner.clean_brand_name)
        
        st.subheader("📊 AFTER:")
        st.metric("Unique Brands (cleaned)", df_clean['brand_cleaned'].nunique())
        
        st.subheader("Data AFTER (Cleaned Brand Names):")
        comparison = df_clean[['brand_name', 'brand_cleaned']].drop_duplicates('brand_name').head(15)
        comparison.columns = ['Original', 'Cleaned']
        st.dataframe(comparison, use_container_width=True)
    
    elif step == "4. Clean Category Names":
        st.subheader("📂 Tahap 4: Clean Category Names")
        st.markdown("""
        **Deskripsi:** Membersihkan dan mapping nama kategori:
        - `*face wash*` → `Face Wash`
        - `*face cream*` → `Face Cream`
        - `*serum*` → `Face Serum`
        - `*body*` → `Body Care`
        - Lainnya → Capitalize
        """)
        
        cleaner = SkincareDataCleaner()
        
        df_clean = df_raw.dropna(subset=['default_category'])
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        st.metric("Unique Categories", df_clean['default_category'].nunique())
        
        st.subheader("Data BEFORE (Original Categories):")
        st.dataframe(df_clean[['product_name', 'default_category']].drop_duplicates('default_category').head(15), use_container_width=True)
        
        # AFTER
        df_clean['category_cleaned'] = df_clean['default_category'].apply(cleaner.clean_category_name)
        
        st.subheader("📊 AFTER:")
        st.metric("Unique Categories (cleaned)", df_clean['category_cleaned'].nunique())
        
        st.subheader("Data AFTER (Cleaned Categories):")
        comparison = df_clean[['default_category', 'category_cleaned']].drop_duplicates('default_category').head(15)
        comparison.columns = ['Original', 'Cleaned']
        st.dataframe(comparison, use_container_width=True)
    
    elif step == "5. Extract Price":
        st.subheader("💰 Tahap 5: Extract Price")
        st.markdown("""
        **Deskripsi:** Mengekstrak harga numerik dari string:
        1. Cari angka dalam string
        2. Konversi ke integer
        3. Batasi range 1.000 - 5.000.000
        """)
        
        cleaner = SkincareDataCleaner()
        
        if 'price_range' in df_raw.columns:
            # BEFORE
            st.subheader("📊 BEFORE:")
            st.metric("Products with price", df_raw['price_range'].notna().sum())
            
            st.subheader("Data BEFORE (Original Price):")
            st.dataframe(df_raw[['product_name', 'price_range']].head(15), use_container_width=True)
            
            # AFTER
            df_raw['price_numeric'] = df_raw['price_range'].apply(cleaner.extract_price)
            
            st.subheader("📊 AFTER:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Min Price", f"Rp {df_raw['price_numeric'].min():,.0f}")
            with col2:
                st.metric("Max Price", f"Rp {df_raw['price_numeric'].max():,.0f}")
            
            st.subheader("Data AFTER (Extracted Price):")
            st.dataframe(df_raw[['product_name', 'price_range', 'price_numeric']].head(15), use_container_width=True)
        else:
            st.info("Kolom 'price_range' tidak ditemukan")
    
    elif step == "6. Filter Rating":
        st.subheader("⭐ Tahap 6: Filter Rating")
        st.markdown("""
        **Deskripsi:** Filter rating yang valid:
        - Rating harus antara 1 dan 5
        - Menghapus rating yang tidak valid
        """)
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
        with col2:
            st.metric("Invalid Ratings", len(df_raw[(df_raw['average_rating'] < 1) | (df_raw['average_rating'] > 5)]))
        
        st.subheader("Data BEFORE (dengan invalid rating):")
        st.dataframe(df_raw[(df_raw['average_rating'] < 1) | (df_raw['average_rating'] > 5)].head(10), use_container_width=True)
        
        # AFTER
        df_filtered = df_raw[(df_raw['average_rating'] >= 1) & (df_raw['average_rating'] <= 5)]
        
        st.subheader("📊 AFTER:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_filtered):,}")
        with col2:
            st.metric("Rows Removed", f"{len(df_raw) - len(df_filtered):,}")
        
        st.subheader("Data AFTER (rating 1-5):")
        st.dataframe(df_filtered[['product_name', 'average_rating']].head(10), use_container_width=True)
    
    elif step == "7. Remove Outliers":
        st.subheader("📊 Tahap 7: Remove Outliers")
        st.markdown("""
        **Deskripsi:** Menghapus outliers:
        - Hapus produk dengan >10.000 reviews
        - Hapus produk dengan wishlist >100.000
        """)
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
        with col2:
            outliers = df_raw[(df_raw['total_reviews'] > 10000) | (df_raw['total_in_wishlist'] > 100000)]
            st.metric("Outliers Found", len(outliers))
        
        st.subheader("Data BEFORE (outliers):")
        st.dataframe(df_raw[(df_raw['total_reviews'] > 10000) | (df_raw['total_in_wishlist'] > 100000)].head(10), use_container_width=True)
        
        # AFTER
        df_filtered = df_raw[
            (df_raw['total_reviews'] <= 10000) & 
            (df_raw['total_in_wishlist'] <= 100000)
        ]
        
        st.subheader("📊 AFTER:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", f"{len(df_filtered):,}")
        with col2:
            st.metric("Rows Removed", f"{len(df_raw) - len(df_filtered):,}")
        
        st.subheader("Data AFTER:")
        st.dataframe(df_filtered[['product_name', 'total_reviews', 'total_in_wishlist']].head(10), use_container_width=True)
    
    elif step == "8. Filter Rare Items":
        st.subheader("🔍 Tahap 8: Filter Rare Items")
        st.markdown("""
        **Deskripsi:** Menghapus brand dan kategori langka:
        - Brand harus memiliki ≥3 produk
        - Kategori harus memiliki ≥5 produk
        """)
        
        brand_counts = df_raw['brand_name'].value_counts()
        category_counts = df_raw['default_category'].value_counts()
        
        valid_brands = brand_counts[brand_counts >= 3].index
        valid_categories = category_counts[category_counts >= 5].index
        
        rare_brands = brand_counts[brand_counts < 3].index
        rare_cats = category_counts[category_counts < 5].index
        
        # BEFORE
        st.subheader("📊 BEFORE:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df_raw):,}")
        with col2:
            st.metric("Total Brands", len(brand_counts))
        with col3:
            st.metric("Total Categories", len(category_counts))
        
        st.subheader("Rare Brands (<3 products):")
        st.dataframe(brand_counts[brand_counts < 3].reset_index().head(10), use_container_width=True)
        
        st.subheader("Rare Categories (<5 products):")
        st.dataframe(category_counts[category_counts < 5].reset_index().head(10), use_container_width=True)
        
        # AFTER
        df_filtered = df_raw[
            (df_raw['brand_name'].isin(valid_brands)) &
            (df_raw['default_category'].isin(valid_categories))
        ]
        
        st.subheader("📊 AFTER:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{len(df_filtered):,}")
        with col2:
            st.metric("Valid Brands", len(valid_brands))
        with col3:
            st.metric("Valid Categories", len(valid_categories))
        
        st.subheader("Data AFTER:")
        st.dataframe(df_filtered[['product_name', 'brand_name', 'default_category']].head(10), use_container_width=True)
    
    elif step == "9. Feature Engineering":
        st.subheader("🔧 Tahap 9: Feature Engineering")
        st.markdown("""
        **Deskripsi:** Membuat fitur untuk ML:
        1. Log transforms (log_reviews, log_wishlist, log_price)
        2. Ratio features (review_wishlist_ratio)
        3. Popularity score
        4. Encode categories
        5. Scale features
        """)
        
        # Show feature list
        features = {
            "Feature": [
                "log_reviews",
                "log_wishlist", 
                "log_price",
                "review_wishlist_ratio",
                "popularity_score",
                "brand_encoded",
                "category_encoded"
            ],
            "Description": [
                "log(total_reviews + 1)",
                "log(total_in_wishlist + 1)",
                "log(price_numeric + 1)",
                "total_reviews / (total_in_wishlist + 1)",
                "normalized popularity",
                "Label encoding brand",
                "Label encoding category"
            ]
        }
        
        st.subheader("📊 Features yang Dibuat:")
        st.table(pd.DataFrame(features))
        
        # Try to load processed data
        try:
            df_processed = pd.read_csv('dataset/processed/skincare_processed.csv')
            st.subheader("📊 Sample Processed Data:")
            st.dataframe(df_processed.head(10), use_container_width=True)
        except:
            st.info("💡 Jalankan `python feature_engineering.py` untuk membuat processed data")