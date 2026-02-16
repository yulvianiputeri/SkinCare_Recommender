import pandas as pd
import numpy as np

print("="*80)
print("STATISTIK DESKRIPTIF DATASET SKINCARE - VERSION FIX")
print("="*80)

# Load dataset
df = pd.read_csv('dataset/processed/skincare_cleaned.csv')

print("\n✓ Dataset berhasil dimuat!")
print(f"  Total produk: {len(df)}")
print(f"  Total kolom: {len(df.columns)}")

# 1. INFO DATASET
print("\n" + "="*80)
print("1. INFORMASI UMUM DATASET")
print("="*80)
print(f"Jumlah total produk: {len(df)}")
print(f"Jumlah atribut: {len(df.columns)}")

print(f"\nKolom yang tersedia:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# 2. CEK & BERSIHKAN DATA NUMERIK
print("\n" + "="*80)
print("2. MEMPROSES DATA NUMERIK")
print("="*80)

# Mapping kolom
cols_map = {
    'average_rating': 'Rating',
    'total_reviews': 'Jumlah Ulasan', 
    'total_in_wishlist': 'Jumlah Wishlist',
    'price_numeric': 'Harga (IDR)'
}

# Cek ketersediaan kolom
print("\nMemeriksa kolom numerik:")
available_cols = {}
for col, label in cols_map.items():
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  ✓ {col} → {label} ({non_null}/{len(df)} values)")
        available_cols[col] = label
    else:
        print(f"  ✗ {col} → TIDAK DITEMUKAN!")

# 3. STATISTIK DESKRIPTIF
print("\n" + "="*80)
print("3. STATISTIK DESKRIPTIF DETAIL")
print("="*80)

stats_data = []

for col, label in available_cols.items():
    data = df[col].dropna()
    
    if len(data) == 0:
        print(f"\n{label}: TIDAK ADA DATA")
        continue
    
    stats = {
        'Atribut': label,
        'Min': data.min(),
        'Max': data.max(),
        'Mean': data.mean(),
        'Median': data.median(),
        'Std Dev': data.std(),
        'Count': len(data)
    }
    stats_data.append(stats)
    
    print(f"\n{label}:")
    print(f"  Min     : {stats['Min']:,.2f}")
    print(f"  Max     : {stats['Max']:,.2f}")
    print(f"  Mean    : {stats['Mean']:,.2f}")
    print(f"  Median  : {stats['Median']:,.2f}")
    print(f"  Std Dev : {stats['Std Dev']:,.2f}")
    print(f"  Count   : {stats['Count']:,}")

# 4. TABEL UNTUK PAPER
print("\n" + "="*80)
print("4. TABEL UNTUK PAPER (COPY-PASTE KE WORD/LATEX)")
print("="*80)

print("\nTabel 2. Statistik Deskriptif Dataset")
print("┌" + "─"*18 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┐")
print("│ Atribut          │ Min        │ Max        │ Mean       │ Median     │ Std Dev    │")
print("├" + "─"*18 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┤")

for stat in stats_data:
    label = stat['Atribut']
    
    if 'Rating' in label:
        # Format untuk rating (1 decimal)
        print(f"│ {label:<16} │ {stat['Min']:>10.1f} │ {stat['Max']:>10.1f} │ {stat['Mean']:>10.2f} │ {stat['Median']:>10.1f} │ {stat['Std Dev']:>10.2f} │")
    elif 'Harga' in label:
        # Format untuk harga (tanpa decimal, dengan koma)
        print(f"│ {label:<16} │ {stat['Min']:>10,.0f} │ {stat['Max']:>10,.0f} │ {stat['Mean']:>10,.0f} │ {stat['Median']:>10,.0f} │ {stat['Std Dev']:>10,.0f} │")
    else:
        # Format untuk reviews/wishlist (tanpa decimal, dengan koma)
        print(f"│ {label:<16} │ {stat['Min']:>10,.0f} │ {stat['Max']:>10,.0f} │ {stat['Mean']:>10,.1f} │ {stat['Median']:>10,.0f} │ {stat['Std Dev']:>10,.1f} │")

print("└" + "─"*18 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┘")

# 5. DISTRIBUSI BRAND & KATEGORI
print("\n" + "="*80)
print("5. DISTRIBUSI BRAND & KATEGORI")
print("="*80)

if 'brand_name_clean' in df.columns:
    print(f"\nJumlah brand unik: {df['brand_name_clean'].nunique()}")
    print("\nTop 10 brand (berdasarkan jumlah produk):")
    top_brands = df['brand_name_clean'].value_counts().head(10)
    for i, (brand, count) in enumerate(top_brands.items(), 1):
        print(f"  {i:2d}. {brand:<30} : {count:>4} produk")
else:
    print("\n✗ Kolom brand_name_clean tidak ditemukan")

if 'category_clean' in df.columns:
    print(f"\nJumlah kategori unik: {df['category_clean'].nunique()}")
    print("\nTop 10 kategori (berdasarkan jumlah produk):")
    top_cats = df['category_clean'].value_counts().head(10)
    for i, (cat, count) in enumerate(top_cats.items(), 1):
        print(f"  {i:2d}. {cat:<30} : {count:>4} produk")
else:
    print("\n✗ Kolom category_clean tidak ditemukan")

# 6. DISTRIBUSI RATING
print("\n" + "="*80)
print("6. DISTRIBUSI RATING")
print("="*80)

if 'average_rating' in df.columns:
    rating_data = df['average_rating'].dropna()
    
    print(f"\nTotal produk dengan rating: {len(rating_data)} ({len(rating_data)/len(df)*100:.1f}%)")
    
    # Distribusi rating dalam bins
    bins = [0, 1, 2, 3, 4, 4.5, 5.0]
    labels = ['0-1', '1-2', '2-3', '3-4', '4-4.5', '4.5-5']
    rating_bins = pd.cut(rating_data, bins=bins, labels=labels, include_lowest=True)
    
    print("\nDistribusi rating:")
    for label in labels:
        count = (rating_bins == label).sum()
        pct = count / len(rating_data) * 100
        print(f"  {label:<8} : {count:>5} produk ({pct:>5.1f}%)")

# 7. ANALISIS POPULARITAS (Long-tail)
print("\n" + "="*80)
print("7. ANALISIS POPULARITAS (Long-tail Distribution)")
print("="*80)

if 'total_reviews' in df.columns:
    reviews = df['total_reviews'].dropna()
    
    q25 = reviews.quantile(0.25)
    q50 = reviews.quantile(0.50)
    q75 = reviews.quantile(0.75)
    q90 = reviews.quantile(0.90)
    q95 = reviews.quantile(0.95)
    
    print(f"\nPersentil ulasan:")
    print(f"  25% produk memiliki ≤ {q25:,.0f} ulasan")
    print(f"  50% produk memiliki ≤ {q50:,.0f} ulasan (median)")
    print(f"  75% produk memiliki ≤ {q75:,.0f} ulasan")
    print(f"  90% produk memiliki ≤ {q90:,.0f} ulasan")
    print(f"  95% produk memiliki ≤ {q95:,.0f} ulasan")
    
    print(f"\nKategori popularitas:")
    long_tail = (reviews <= q75).sum()
    popular = ((reviews > q75) & (reviews <= q95)).sum()
    very_popular = (reviews > q95).sum()
    
    print(f"  Long-tail (≤Q75)      : {long_tail:>5} produk ({long_tail/len(reviews)*100:>5.1f}%)")
    print(f"  Popular (Q75-Q95)     : {popular:>5} produk ({popular/len(reviews)*100:>5.1f}%)")
    print(f"  Very Popular (>Q95)   : {very_popular:>5} produk ({very_popular/len(reviews)*100:>5.1f}%)")

# 8. MISSING VALUES 
print("\n" + "="*80)
print("8. MISSING VALUES (KOLOM PENTING)")
print("="*80)

important_cols = ['brand_name', 'product_name', 'average_rating', 'total_reviews', 
                    'total_in_wishlist', 'price_numeric', 'default_category']

print("\nKolom penting dengan missing values:")
has_missing = False
for col in important_cols:
    if col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            pct = missing / len(df) * 100
            print(f"  {col:<25} : {missing:>5} ({pct:>5.1f}%)")
            has_missing = True

if not has_missing:
    print("  ✓ Tidak ada missing values pada kolom penting!")

# 9. TRAIN-TEST SPLIT
print("\n" + "="*80)
print("9. PEMBAGIAN DATA (TRAIN-TEST SPLIT)")
print("="*80)

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size

print(f"\nPembagian dataset (80:20):")
print(f"  Total dataset    : {len(df):>5} produk")
print(f"  Training set     : {train_size:>5} produk (80%)")
print(f"  Test set         : {test_size:>5} produk (20%)")

print(f"\nCatatan:")
print(f"  Stratified sampling berdasarkan kategori produk direkomendasikan")
print(f"  untuk memastikan representasi yang seimbang pada kedua set.")

# 10. SUMMARY UNTUK PAPER
print("\n" + "="*80)
print("10. SUMMARY UNTUK SECTION DATASET DI PAPER")
print("="*80)

print(f"""
Dataset yang digunakan dalam penelitian ini terdiri dari {len(df)} produk 
skincare yang diperoleh dari platform Sociolla melalui repositori Kaggle. 
Dataset ini merepresentasikan karakteristik pasar e-commerce kecantikan 
Indonesia dengan cakupan {df['brand_name_clean'].nunique() if 'brand_name_clean' in df.columns else 'XXX'}+ merek dari berbagai segmen harga 
dan {df['category_clean'].nunique() if 'category_clean' in df.columns else 'XX'} kategori produk yang beragam.

Untuk evaluasi model, dataset dibagi menjadi training set dan test set 
dengan proporsi 80:20 menggunakan stratified random sampling berdasarkan 
kategori produk untuk memastikan representasi yang seimbang pada kedua set. 
Training set ({train_size} produk) digunakan untuk melatih model TF-IDF, 
SVD, dan Random Forest, sementara test set ({test_size} produk) digunakan 
untuk evaluasi performa sistem ensemble.
""")

print("\n" + "="*80)
print("SELESAI!")
print("="*80)
print("\n✓ Silakan copy-paste Tabel 2 di atas ke dalam paper Anda!")
print("✓ Simpan output ini untuk referensi!")