import pandas as pd
import numpy as np

print("="*80)
print("TABEL PERBANDINGAN BEFORE-AFTER PREPROCESSING")
print("="*80)

# Load dataset
df_raw = pd.read_csv('dataset/sociolla.csv')
df_clean = pd.read_csv('dataset/processed/skincare_cleaned.csv')

print("\n✓ Dataset berhasil dimuat!")
print(f"  Raw dataset: {len(df_raw)} produk")
print(f"  Cleaned dataset: {len(df_clean)} produk")

# Hitung statistik
stats = {
    'Atribut': [],
    'Before': [],
    'After': [],
    'Perubahan': []
}

# 1. Jumlah Produk
stats['Atribut'].append('Jumlah Produk')
stats['Before'].append(len(df_raw))
stats['After'].append(len(df_clean))
stats['Perubahan'].append(len(df_clean) - len(df_raw))

# 2. Jumlah Brand
before_brands = df_raw['brand_name'].nunique() if 'brand_name' in df_raw.columns else 0
after_brands = df_clean['brand_name_clean'].nunique() if 'brand_name_clean' in df_clean.columns else df_clean['brand_name'].nunique()
stats['Atribut'].append('Jumlah Brand Unik')
stats['Before'].append(before_brands)
stats['After'].append(after_brands)
stats['Perubahan'].append(after_brands - before_brands)

# 3. Jumlah Kategori
before_cats = df_raw['default_category'].nunique() if 'default_category' in df_raw.columns else 0
after_cats = df_clean['category_clean'].nunique() if 'category_clean' in df_clean.columns else df_clean['default_category'].nunique()
stats['Atribut'].append('Jumlah Kategori')
stats['Before'].append(before_cats)
stats['After'].append(after_cats)
stats['Perubahan'].append(after_cats - before_cats)

# 4. Produk dengan Rating
before_rating = df_raw['average_rating'].notna().sum() if 'average_rating' in df_raw.columns else 0
after_rating = df_clean['average_rating'].notna().sum() if 'average_rating' in df_clean.columns else 0
stats['Atribut'].append('Produk dengan Rating')
stats['Before'].append(before_rating)
stats['After'].append(after_rating)
stats['Perubahan'].append(after_rating - before_rating)

# 5. Missing Values (rata-rata)
before_missing = df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns)) * 100
after_missing = df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns)) * 100
stats['Atribut'].append('Missing Values (%)')
stats['Before'].append(before_missing)
stats['After'].append(after_missing)
stats['Perubahan'].append(after_missing - before_missing)

# 6. Outlier Reviews (>10,000)
before_outlier_rev = (df_raw['total_reviews'] > 10000).sum() if 'total_reviews' in df_raw.columns else 0
after_outlier_rev = (df_clean['total_reviews'] > 10000).sum() if 'total_reviews' in df_clean.columns else 0
stats['Atribut'].append('Outlier Reviews (>10K)')
stats['Before'].append(before_outlier_rev)
stats['After'].append(after_outlier_rev)
stats['Perubahan'].append(after_outlier_rev - before_outlier_rev)

# 7. Outlier Wishlist (>5,000)
before_outlier_wish = (df_raw['total_in_wishlist'] > 5000).sum() if 'total_in_wishlist' in df_raw.columns else 0
after_outlier_wish = (df_clean['total_in_wishlist'] > 5000).sum() if 'total_in_wishlist' in df_clean.columns else 0
stats['Atribut'].append('Outlier Wishlist (>5K)')
stats['Before'].append(before_outlier_wish)
stats['After'].append(after_outlier_wish)
stats['Perubahan'].append(after_outlier_wish - before_outlier_wish)

# 8. Rating Range
if 'average_rating' in df_raw.columns and 'average_rating' in df_clean.columns:
    before_rating_min = df_raw['average_rating'].min()
    before_rating_max = df_raw['average_rating'].max()
    after_rating_min = df_clean['average_rating'].min()
    after_rating_max = df_clean['average_rating'].max()
    
    stats['Atribut'].append('Rating Min')
    stats['Before'].append(before_rating_min)
    stats['After'].append(after_rating_min)
    stats['Perubahan'].append(after_rating_min - before_rating_min)
    
    stats['Atribut'].append('Rating Max')
    stats['Before'].append(before_rating_max)
    stats['After'].append(after_rating_max)
    stats['Perubahan'].append(after_rating_max - before_rating_max)

df_stats = pd.DataFrame(stats)

# Tampilkan statistik detail
print("\n" + "="*80)
print("STATISTIK DETAIL")
print("="*80)
for idx, row in df_stats.iterrows():
    print(f"\n{row['Atribut']}:")
    print(f"  Before: {row['Before']:,.2f}" if isinstance(row['Before'], float) else f"  Before: {row['Before']:,}")
    print(f"  After:  {row['After']:,.2f}" if isinstance(row['After'], float) else f"  After:  {row['After']:,}")
    
    change = row['Perubahan']
    if isinstance(change, float):
        print(f"  Change: {change:+,.2f}")
    else:
        print(f"  Change: {change:+,}")

print("\n" + "="*80)
print("TABEL 3 UNTUK PAPER (COPY-PASTE INI)")
print("="*80)

print("\nTabel 3. Perbandingan Dataset Before-After Preprocessing")
print("┌" + "─"*27 + "┬" + "─"*13 + "┬" + "─"*13 + "┬" + "─"*13 + "┐")
print("│ Atribut                   │ Before      │ After       │ Perubahan   │")
print("├" + "─"*27 + "┼" + "─"*13 + "┼" + "─"*13 + "┼" + "─"*13 + "┤")

for idx, row in df_stats.iterrows():
    attr = row['Atribut']
    before = row['Before']
    after = row['After']
    change = row['Perubahan']
    
    # Format berdasarkan tipe data
    if 'Missing' in attr or 'Rating' in attr:
        # Format dengan 2 decimal untuk persentase dan rating
        print(f"│ {attr:<25} │ {before:>11.2f} │ {after:>11.2f} │ {change:>+11.2f} │")
    else:
        # Format integer dengan koma
        before_str = f"{int(before):,}" if not pd.isna(before) else "N/A"
        after_str = f"{int(after):,}" if not pd.isna(after) else "N/A"
        change_str = f"{int(change):+,}" if not pd.isna(change) else "N/A"
        print(f"│ {attr:<25} │ {before_str:>11} │ {after_str:>11} │ {change_str:>11} │")

print("└" + "─"*27 + "┴" + "─"*13 + "┴" + "─"*13 + "┴" + "─"*13 + "┘")

# Interpretasi
print("\n" + "="*80)
print("INTERPRETASI")
print("="*80)

removed = len(df_raw) - len(df_clean)
removed_pct = (removed / len(df_raw)) * 100

print(f"""
RINGKASAN PREPROCESSING:
- {removed:,} produk dihapus ({removed_pct:.1f}% dari dataset awal)
- Outlier reviews dan wishlist berhasil difilter
- Missing values berkurang dari {before_missing:.1f}% menjadi {after_missing:.1f}%
- Dataset menjadi lebih bersih dan konsisten

UNTUK PAPER, TULISKAN:
"Proses preprocessing menghapus {removed:,} produk ({removed_pct:.1f}%) yang memiliki 
outlier ekstrem atau missing values pada kolom penting, menghasilkan dataset 
final dengan {len(df_clean):,} produk yang siap untuk modeling (Tabel 3)."
""")

print("\n" + "="*80)
print("SELESAI!")
print("="*80)