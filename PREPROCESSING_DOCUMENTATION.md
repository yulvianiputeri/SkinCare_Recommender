# 📋 Dokumentasi Preprocessing Data

## Overview

Dokumen ini menjelaskan tahapan preprocessing data dari dataset Sociolla untuk skincare products.

---

## 📊 Ringkasan Hasil

| Tahap | Proses | Before | After | Removed |
|-------|--------|--------|-------|---------|
| 1 | Load Data | - | 7,636 | - |
| 2 | Drop Missing | 7,636 | 7,636 | 0 |
| 3 | Clean Brand | 321 | 321 | 0 |
| 4 | Clean Category | 195 | 177 | 18 |
| 5 | Extract Price | - | - | - |
| 6 | Filter Rating | 7,636 | 5,790 | 1,846 |
| 7 | Remove Outliers | 5,790 | 5,779 | 11 |
| 8 | Filter Rare | 5,779 | 5,613 | 166 |

**Final: 7,636 → 5,613 rows (73.5% retained)**

---

## Detail Tiap Tahap

### Tahap 1: Load Data

**Deskripsi:** Memuat data asli dari file CSV ke memory.

**Input:** `dataset/sociolla.csv`

**Output:** DataFrame dengan 7,636 baris dan 19 kolom

**Kolom-kolom yang ada:**
- product_name
- brand_name
- default_category
- average_rating
- total_reviews
- total_in_wishlist
- price_range
- dll

---

### Tahap 2: Drop Missing Values

**Deskripsi:** Menghapus baris yang tidak memiliki data essential.

**Kriteria:**
- `product_name` tidak boleh kosong
- `brand_name` tidak boleh kosong
- `average_rating` tidak boleh kosong

**Hasil:**
| Metric | Value |
|--------|-------|
| Before | 7,636 |
| After | 7,636 |
| Removed | 0 |

**Catatan:** Tidak ada yang dihapus karena semua baris sudah memiliki data essential.

---

### Tahap 3: Clean Brand Names

**Deskripsi:** Membersihkan nama brand agar konsisten.

**Proses Cleaning:**
1. Hapus prefix numerik (contoh: `796_3ce` → `3ce` → `Unknown`)
2. Ambil bagian yang meaningful
3. Capitalize huruf pertama
4. Batasi max 20 karakter

**Contoh Cleaning:**

| Original | After Cleaning |
|----------|---------------|
| `796_3ce` | `Unknown` |
| `751_abib` | `Abib` |
| `730_acwell` | `Acwell` |
| `15654_adara-cosmetics` | `Adaracosmetics` |
| `584_aeris-beaute` | `Aerisbeaute` |
| `15703_alatte` | `Alatte` |
| `15542_alchemist-fragrance` | `Alchemistfragrance` |
| `1350_allglows` | `Allglows` |
| `15700_amuse` | `Amuse` |
| `2019_anessa` | `Anessa` |

**Kenapa ada yang jadi "Unknown"?**
- Brand dengan prefix numerik saja (contoh: `796_3ce` → setelah hapus angka, tidak tersisa karakter yang meaningful)

**Hasil:**
| Metric | Value |
|--------|-------|
| Unique Before | 321 |
| Unique After | 321 |
| Removed | 0 |

---

### Tahap 4: Clean Category Names

**Deskripsi:** Membersihkan dan mapping nama kategori agar konsisten.

**Proses Cleaning:**
1. Hapus karakter tidak perlu (kurung kurawal, dll)
2. Mapping kategori spesifik:
   - `*face wash*` → `Face Wash`
   - `*face cream*` → `Face Cream`
   - `*serum*` → `Face Serum`
   - `*body*` → `Body Care`
3. Capitalize huruf pertama
4. Batasi max 30 karakter

**Contoh Cleaning:**

| Original | After Cleaning |
|----------|---------------|
| `Eyeshadow` | `Eyeshadow` |
| `Lip Cream` | `Lip Cream` |
| `Lipstick` | `Lipstick` |
| `Blush` | `Blush` |
| `Lip Tint` | `Lip Tint` |
| `Eyebrows` | `Eyebrows` |
| `Foundation` | `Foundation` |
| `Mascara` | `Mascara` |
| `Pressed Powder` | `Pressed Powder` |
| `Eyeliner` | `Eyeliner` |

**Hasil:**
| Metric | Value |
|--------|-------|
| Unique Before | 195 |
| Unique After | 177 |
| Removed | 18 (duplicates/tidak valid) |

---

### Tahap 5: Extract Price

**Deskripsi:** Mengekstrak harga numerik dari string.

**Proses:**
1. Cari angka dalam string (regex: `[\d,]+`)
2. Hapus koma
3. Konversi ke integer
4. Batasi range 1.000 - 5.000.000

**Contoh:**

| Original (price_range) | Extracted (price_numeric) |
|---------------------|-----------------------|
| `100000` | 100,000 |
| `150000 - 200000` | 150,000 |
| `Rp 100.000` | 100,000 |

**Hasil:**
| Metric | Value |
|--------|-------|
| Min Price | Rp 1,000 |
| Max Price | Rp 1,000 |
| Mean Price | Rp 1,000 |

**Catatan:** Pada dataset ini, semua price_range sudah bernilai 1,000 (default karena kolom tidak ada/tidak valid).

---

### Tahap 6: Filter Rating

**Deskripsi:** Filter rating yang valid.

**Kriteria:**
- Rating harus antara 1 dan 5
- Menghapus rating yang tidak valid (< 1 atau > 5)

**Hasil:**
| Metric | Value |
|--------|-------|
| Before | 7,636 |
| After | 5,790 |
| Removed | 1,846 |

**Kenapa dihapus banyak?**
- Banyak produk dengan rating 0 atau null
- Beberapa produk dengan rating > 5 (tidak valid)

---

### Tahap 7: Remove Outliers

**Deskripsi:** Menghapus produk dengan nilai ekstrem.

**Kriteria:**
- `total_reviews` ≤ 10,000
- `total_in_wishlist` ≤ 100,000

**Hasil:**
| Metric | Value |
|--------|-------|
| Before | 5,790 |
| After | 5,779 |
| Removed | 11 |

---

### Tahap 8: Filter Rare Items

**Deskripsi:** Menghapus brand dan kategori yang terlalu sedikit produknya.

**Kriteria:**
- Brand harus memiliki ≥ 3 produk
- Kategori harus memiliki ≥ 5 produk

**Hasil:**
| Metric | Value |
|--------|-------|
| Before | 5,779 |
| After | 5,613 |
| Removed | 166 |
| Valid Brands | 282 |
| Valid Categories | 114 |

---

## 📈 Kesimpulan

### Data Retention

```
Raw Data:     7,636 rows
Cleaned:     5,613 rows
Retention:   73.5%
```

### Distribusi Akhir

| Metric | Value |
|--------|-------|
| Total Produk | 5,613 |
| Total Brands | 282 |
| Total Categories | 114 |
| Rata-rata Rating | 4.64 |

---

## Cara Menjalankan

```bash
# Jalankan script preprocessing
python3 show_preprocessing.py

# Atau jalankan full pipeline
python3 data_pipeline.py

# Atau buka di Streamlit
streamlit run app.py
```

---

## File Input/Output

| File | Deskripsi |
|------|----------|
| Input | `dataset/sociolla.csv` |
| Output | `dataset/processed/skincare_cleaned.csv` |
| Output | `dataset/processed/skincare_processed.csv` |

---

*Last Updated: May 2026*