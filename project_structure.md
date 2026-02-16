# 📁 PROJECT STRUCTURE - Clean & Organized

## 🎯 Overview

This document explains the **essential files** in the project after cleanup.

---

## 📂 File Structure (11 Essential Files)

```
skincare-recommender/
│
├── 🎨 FRONTEND & UI
│   ├── app.py                      # Main Streamlit application
│   ├── analytics.py                # Analytics dashboard
│   └── utils.py                    # Utility functions (formatting, CSS)
│
├── 🧠 CORE SYSTEM
│   ├── ensemble_system.py          # Main hybrid recommender system
│   ├── tfidf_model.py              # TF-IDF content-based filtering
│   ├── svd_model.py                # SVD collaborative filtering
│   └── rf_model_improved.py        # Random Forest (7 features, R²=0.624)
│
├── 🔧 DATA PROCESSING
│   ├── preprocessing_data.py       # Data cleaning
│   ├── feature_engineering.py      # Feature creation
│   └── data_pipeline.py            # Complete pipeline
│
└── 📦 CONFIGURATION
    └── requirements.txt            # Python dependencies
```

---

## 📋 File Descriptions

### 🎨 **Frontend & UI**

#### **app.py** (Main Application)
- **Purpose:** Streamlit web interface
- **Features:**
  - Product recommendations
  - Similar product finder
  - Rating prediction
  - Analytics dashboard
- **Usage:** `streamlit run app.py`

#### **analytics.py** (Dashboard)
- **Purpose:** Data visualization & insights
- **Features:**
  - Rating distribution
  - Brand/category charts
  - Top products
  - Key metrics
- **Used by:** app.py

#### **utils.py** (Utilities)
- **Purpose:** Helper functions
- **Features:**
  - Price formatting
  - Recommendation explanations
  - Product filtering
  - CSS styles
- **Used by:** app.py

---

### 🧠 **Core System**

#### **ensemble_system.py** (Main System)
- **Purpose:** Hybrid recommender coordinator
- **Components:**
  - TF-IDF model
  - SVD model
  - RF model
  - Meta-learner (ensemble)
- **Key Methods:**
  - `train_all_models()` - Train all components
  - `get_recommendations()` - Get hybrid recommendations
  - `get_ensemble_score()` - Get ensemble prediction

#### **tfidf_model.py** (Content-Based)
- **Purpose:** TF-IDF content similarity
- **Features:**
  - Text vectorization (1000 features)
  - Cosine similarity
  - Product name + brand + category
- **Performance:** R² = 0.803 (best individual model)

#### **svd_model.py** (Collaborative)
- **Purpose:** SVD matrix factorization
- **Features:**
  - 30 latent components
  - Synthetic user interactions
  - User-item matrix
- **Note:** Currently has issues (R² negative), may be excluded

#### **rf_model_improved.py** (Metadata Prediction)
- **Purpose:** Rating prediction from metadata
- **Features:** 7 features
  1. brand_encoded
  2. category_encoded
  3. log_reviews
  4. log_wishlist
  5. log_price
  6. popularity_score
  7. review_wishlist_ratio
- **Performance:** R² = 0.624

---

### 🔧 **Data Processing**

#### **preprocessing_data.py** (Data Cleaning)
- **Purpose:** Clean raw data
- **Steps:**
  1. Load raw CSV
  2. Clean brand/category names
  3. Extract price
  4. Filter outliers
  5. Remove invalid data
- **Output:** `dataset/processed/skincare_cleaned.csv`

#### **feature_engineering.py** (Feature Creation)
- **Purpose:** Create ML features
- **Steps:**
  1. Log transforms
  2. Ratio features
  3. Popularity score
  4. Encoding categories
  5. Scaling
- **Output:** `dataset/processed/skincare_processed.csv`

#### **data_pipeline.py** (Complete Pipeline)
- **Purpose:** Run full preprocessing pipeline
- **Steps:**
  1. Data cleaning
  2. Feature engineering
  3. Create sample (if dataset > 8000)
- **Usage:** `python data_pipeline.py`

---

## 🚀 Usage Guide

### **1. First Time Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Run data pipeline (if you have raw data)
python data_pipeline.py

# Train models
python ensemble_system.py
```

### **2. Run Application**

```bash
# Start Streamlit app
streamlit run app.py
```

### **3. Using Individual Components**

```python
# Example: Use TF-IDF only
from tfidf_model import TFIDFContentModel

tfidf = TFIDFContentModel()
tfidf.train()
similar_products = tfidf.get_similar(product_idx=0, n=10)
```

```python
# Example: Use ensemble system
from ensemble_system import EnsembleHybridSystem

system = EnsembleHybridSystem()
system.load_trained_models()  # or train_all_models()
recommendations = system.get_recommendations(n_recommendations=10)
```

---

## 📊 Model Performance

| Model | R² | MAE | RMSE | Notes |
|-------|-----|-----|------|-------|
| **TF-IDF Only** | 0.803 | 0.116 | 0.145 | Best individual |
| **RF Only** | 0.624 | 0.100 | 0.168 | Improved from 0.116 |
| **SVD Only** | -11.18 | 1.119 | 1.140 | Has issues |
| **Ensemble** | 0.688 | 0.088 | 0.183 | Hybrid approach |

---

## 🗂️ Data Files

### **Input:**
- `dataset/sociolla.csv` - Raw dataset (7,500+ products)

### **Processed:**
- `dataset/processed/skincare_cleaned.csv` - Cleaned data
- `dataset/processed/skincare_processed.csv` - With features (used by models)
- `dataset/processed/skincare_ml_features.csv` - ML features only

### **Models:**
- `models/tfidf_model.pkl` - TF-IDF model
- `models/svd_model.pkl` - SVD model
- `models/rf_model.pkl` - Random Forest model
- `models/ensemble_system.pkl` - Ensemble meta-learner

---

## 🔄 Workflow

```
1. Raw Data (sociolla.csv)
   ↓
2. preprocessing_data.py → skincare_cleaned.csv
   ↓
3. feature_engineering.py → skincare_processed.csv
   ↓
4. Train Models:
   - tfidf_model.py
   - svd_model.py
   - rf_model_improved.py
   ↓
5. ensemble_system.py → Combine all models
   ↓
6. app.py → User interface
```

---

## 🧹 Archived Files

The following files have been archived (moved to `archive/` folder):

- `hybrid_model.py` - Just aliases, not needed
- `rf_model.py` - Old version (replaced by rf_model_improved.py)
- `apply_patch.py` - One-time use script
- `quick_fix.py` - One-time use script
- `baseline_comparison.py` - Already executed
- `visualization.py` - Already executed
- `run_all_revisions.py` - Already executed
- `README.md` - Old documentation

**To restore:** `mv archive/[filename] .`

---

## 📝 Notes

### **Why RF Improved?**
- Original: 4 features, R² = 0.116 ❌
- Improved: 7 features, R² = 0.624 ✅
- Added: log_price, popularity_score, review_wishlist_ratio

### **Why SVD has issues?**
- Uses synthetic user interactions
- Not based on real user data
- May be excluded from final ensemble

### **Best Individual Model?**
- **TF-IDF** (R² = 0.803)
- Strong content-based similarity
- Works well for skincare products

---

## 🎯 For Paper/Thesis

### **Key Points:**
1. ✅ Hybrid system: TF-IDF + RF (SVD optional)
2. ✅ RF improved: 437% better (0.116 → 0.624)
3. ✅ Multiple metrics: R², MAE, RMSE
4. ✅ Clean codebase: 11 essential files
5. ✅ Production-ready: Streamlit app

### **Results to Report:**
- TF-IDF: R² = 0.803 (best)
- RF: R² = 0.624 (improved)
- Ensemble: R² = 0.688 (hybrid)

---

## 🆘 Troubleshooting

### **"Module not found"**
```bash
pip install -r requirements.txt
```

### **"No data found"**
```bash
# Run data pipeline first
python data_pipeline.py
```

### **"Model not trained"**
```bash
# Train models
python ensemble_system.py
```

### **"Streamlit error"**
```bash
# Make sure all dependencies installed
pip install streamlit plotly
streamlit run app.py
```

---

## 📞 Quick Commands

```bash
# Cleanup project
python cleanup.py

# Run pipeline
python data_pipeline.py

# Train models
python ensemble_system.py

# Start app
streamlit run app.py

# Test TF-IDF
python tfidf_model.py

# Test RF
python rf_model_improved.py
```

---

**Last Updated:** 2026-01-31  
**Status:** ✅ Production Ready  
**Files:** 11 Essential Files