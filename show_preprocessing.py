"""Show Preprocessing Steps - Standalone Script"""
import pandas as pd
from preprocessing_data import SkincareDataCleaner

# Load raw data
print("=" * 60)
print("📥 TAHAP 1: LOAD DATA")
print("=" * 60)
df_raw = pd.read_csv('dataset/sociolla.csv')
print(f"Total Rows: {len(df_raw):,}")
print(f"Total Columns: {len(df_raw.columns)}")
print(f"Missing Values: {df_raw.isna().sum().sum()}")
print("\nSample:")
print(df_raw.head(3))

# Initialize cleaner
cleaner = SkincareDataCleaner()

# Drop missing
print("\n" + "=" * 60)
print("🗑️ TAHAP 2: DROP MISSING VALUES")
print("=" * 60)
essential_cols = ['product_name', 'brand_name', 'average_rating']
df = df_raw.dropna(subset=essential_cols).copy()
print(f"BEFORE: {len(df_raw):,} rows")
print(f"AFTER:  {len(df):,} rows")
print(f"Removed: {len(df_raw) - len(df):,} rows")

# Clean brand
print("\n" + "=" * 60)
print("🧼 TAHAP 3: CLEAN BRAND NAMES")
print("=" * 60)
df['brand_name'] = df['brand_name'].apply(cleaner.clean_brand_name)
print(f"Unique BEFORE: {df_raw['brand_name'].nunique()}")
print(f"Unique AFTER:  {df['brand_name'].nunique()}")
print(f"Unknown brands: {(df['brand_name'] == 'Unknown').sum()}")
print("\nContoh:")
sample = df[['brand_name']].drop_duplicates('brand_name').head(10)
for brand in sample['brand_name']:
    print(f"  {brand}")

# Clean category
print("\n" + "=" * 60)
print("📂 TAHAP 4: CLEAN CATEGORY NAMES")
print("=" * 60)
df['default_category'] = df['default_category'].apply(cleaner.clean_category_name)
print(f"Unique BEFORE: {df_raw['default_category'].nunique()}")
print(f"Unique AFTER:  {df['default_category'].nunique()}")

# Extract price
print("\n" + "=" * 60)
print("💰 TAHAP 5: EXTRACT PRICE")
print("=" * 60)
df['price_numeric'] = df['price_range'].apply(cleaner.extract_price)
print(f"Min: Rp {df['price_numeric'].min():,.0f}")
print(f"Max: Rp {df['price_numeric'].max():,.0f}")
print(f"Mean: Rp {df['price_numeric'].mean():,.0f}")

# Filter rating
print("\n" + "=" * 60)
print("⭐ TAHAP 6: FILTER RATING (1-5)")
print("=" * 60)
df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
before = len(df)
df = df[(df['average_rating'] >= 1) & (df['average_rating'] <= 5)]
print(f"BEFORE: {before:,} rows")
print(f"AFTER:  {len(df):,} rows")
print(f"Removed: {before - len(df):,} rows")
print(f"Unknown brands remaining: {(df['brand_name'] == 'Unknown').sum()}")

# Remove outliers
print("\n" + "=" * 60)
print("📊 TAHAP 7: REMOVE OUTLIERS")
print("=" * 60)
df['total_reviews'] = pd.to_numeric(df['total_reviews'], errors='coerce').fillna(0)
df['total_in_wishlist'] = pd.to_numeric(df['total_in_wishlist'], errors='coerce').fillna(0)
before = len(df)
df = df[(df['total_reviews'] <= 10000) & (df['total_in_wishlist'] <= 100000)]
print(f"BEFORE: {before:,} rows")
print(f"AFTER:  {len(df):,} rows")
print(f"Removed: {before - len(df):,} rows")
print(f"Unknown brands remaining: {(df['brand_name'] == 'Unknown').sum()}")

# Filter rare
print("\n" + "=" * 60)
print("🔍 TAHAP 8: FILTER RARE ITEMS")
print("=" * 60)
brand_counts = df['brand_name'].value_counts()
category_counts = df['default_category'].value_counts()
valid_brands = brand_counts[brand_counts >= 3].index
valid_categories = category_counts[category_counts >= 5].index

before = len(df)
df = df[(df['brand_name'].isin(valid_brands)) & (df['default_category'].isin(valid_categories))]
print(f"BEFORE: {before:,} rows")
print(f"AFTER:  {len(df):,} rows")
print(f"Removed: {before - len(df):,} rows")
print(f"Valid Brands: {len(valid_brands)}")
print(f"Valid Categories: {len(valid_categories)}")
print(f"Unknown brands remaining: {(df['brand_name'] == 'Unknown').sum()}")

# Remove Unknown brand
print("\n" + "=" * 60)
print("❌ TAHAP 9: REMOVE UNKNOWN BRAND")
print("=" * 60)
before = len(df)
unknown_count = (df['brand_name'] == 'Unknown').sum()
df = df[df['brand_name'] != 'Unknown']
print(f"BEFORE: {before:,} rows")
print(f"Unknown brands found: {unknown_count}")
print(f"AFTER:  {len(df):,} rows")
print(f"Removed: {before - len(df):,} rows")

# Summary
print("\n" + "=" * 60)
print("📈 RINGKASAN")
print("=" * 60)
print(f"Raw Data:     {len(df_raw):,} rows")
print(f"Cleaned:     {len(df):,} rows  (setelah hapus Unknown)")
print(f"Brands:      {df['brand_name'].nunique()}")
print(f"Categories:  {df['default_category'].nunique()}")
print(f"Avg Rating:  {df['average_rating'].mean():.2f}")