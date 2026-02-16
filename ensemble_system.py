import pandas as pd
import numpy as np
import os
import pickle
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Import individual models
from tfidf_model import TFIDFContentModel
from svd_model import SVDCollaborativeModel
from rf_model_improved import RandomForestModel

class EnsembleHybridSystem:
    """Main Hybrid Recommender - TF-IDF + SVD + Random Forest Ensemble"""
    
    def __init__(self):
        # Initialize component models
        self.tfidf_model = TFIDFContentModel()
        self.svd_model = SVDCollaborativeModel()
        self.rf_model = RandomForestModel()
        
        # Ensemble meta-learner
        self.meta_learner = RandomForestRegressor(n_estimators=30, random_state=42)
        self.meta_scaler = StandardScaler()
        
        # Status
        self.is_trained = False
        self.df = None
        
        os.makedirs('models', exist_ok=True)
    
    def load_data(self):
        paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
        for path in paths:
            try:
                self.df = pd.read_csv(path)
                print(f"📥 Loaded: {len(self.df):,} products from {path}")
                return True
            except:
                continue
        return False
    
    def train_all_models(self):
        """Train all component models and ensemble"""
        print("🚀 TRAINING ENSEMBLE HYBRID SYSTEM")
        print("="*50)
        
        if not self.load_data():
            raise ValueError("Cannot load data!")
        
        results = {}
        
        # Train individual models
        print("\n📋 Step 1/4: TF-IDF Content Model")
        self.tfidf_model.train()
        
        print("\n📋 Step 2/4: SVD Collaborative Model")
        self.svd_model.train()
        
        print("\n📋 Step 3/4: Random Forest Model")
        rf_results = self.rf_model.train()
        results.update(rf_results)
        
        print("\n📋 Step 4/4: Ensemble Meta-Learner")
        ensemble_results = self.train_ensemble()
        results.update(ensemble_results)
        
        self.is_trained = True
        self.save_system()
        
        print("\n✅ ENSEMBLE SYSTEM READY!")
        print(f"📊 Results: {results}")
        return results
    
    def train_ensemble(self):
        """Train ensemble meta-learner"""
        print("🧠 Training ensemble meta-learner...")
        
        # Sample for training
        sample_size = min(800, len(self.df))
        sample_indices = np.random.choice(len(self.df), sample_size, replace=False)
        
        features = []
        targets = []
        
        for idx in sample_indices:
            tfidf_score = self.tfidf_model.get_score(idx)
            svd_score = self.svd_model.get_score(idx)
            rf_score = self.rf_model.get_score(idx)
            
            features.append([tfidf_score, svd_score, rf_score])
            targets.append(self.df.iloc[idx]['average_rating'])
        
        # Train meta-learner with proper feature names
        feature_names = ['tfidf_score', 'svd_score', 'rf_score']
        X = pd.DataFrame(features, columns=feature_names)
        y = np.array(targets)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = pd.DataFrame(
            self.meta_scaler.fit_transform(X_train), 
            columns=feature_names
        )
        X_val_scaled = pd.DataFrame(
            self.meta_scaler.transform(X_val), 
            columns=feature_names
        )
        
        self.meta_learner.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.meta_learner.predict(X_val_scaled)
        ensemble_r2 = r2_score(y_val, y_pred)
        
        print(f"🎯 Ensemble R²: {ensemble_r2:.3f}")
        return {'ensemble_r2': ensemble_r2}
    
    def get_ensemble_score(self, product_idx, user_preferences=None):
        """Get ensemble prediction for a product"""
        if not self.is_trained:
            return {'ensemble_score': 0.5, 'tfidf_score': 0.5, 'svd_score': 0.5, 'rf_score': 0.5}
        
        try:
            # Get individual scores
            tfidf_score = self.tfidf_model.get_score(product_idx, user_preferences)
            svd_score = self.svd_model.get_score(product_idx)
            rf_score = self.rf_model.get_score(product_idx)
            
            # Ensemble prediction with proper feature names
            feature_names = ['tfidf_score', 'svd_score', 'rf_score']
            features = pd.DataFrame([[tfidf_score, svd_score, rf_score]], columns=feature_names)
            features_scaled = pd.DataFrame(
                self.meta_scaler.transform(features), 
                columns=feature_names
            )
            
            ensemble_rating = self.meta_learner.predict(features_scaled)[0]
            ensemble_score = np.clip(ensemble_rating / 5.0, 0.1, 1.0)
            
            return {
                'ensemble_score': ensemble_score,
                'tfidf_score': tfidf_score,
                'svd_score': svd_score,
                'rf_score': rf_score
            }
        except:
            # Fallback
            return {'ensemble_score': 0.5, 'tfidf_score': 0.5, 'svd_score': 0.5, 'rf_score': 0.5}
    
    def get_recommendations(self, user_preferences=None, n_recommendations=10):
        """Get hybrid recommendations"""
        if not self.is_trained:
            raise ValueError("Models not trained!")
        
        # Filter candidates
        candidates = self.df.copy()
        
        if user_preferences:
            if user_preferences.get('brands'):
                candidates = candidates[candidates['brand_name'].isin(user_preferences['brands'])]
            if user_preferences.get('categories'):
                candidates = candidates[candidates['default_category'].isin(user_preferences['categories'])]
            if user_preferences.get('min_rating'):
                candidates = candidates[candidates['average_rating'] >= user_preferences['min_rating']]
        
        if candidates.empty:
            candidates = self.df.copy()
        
        # Limit for performance
        if len(candidates) > 1000:
            candidates = candidates.sample(n=1000, random_state=42)
        
        # Score candidates
        recommendations = []
        for idx in candidates.index:
            try:
                original_idx = self.df.index.get_loc(idx)
                scores = self.get_ensemble_score(original_idx, user_preferences)
                
                product_info = candidates.loc[idx].copy()
                for key, value in scores.items():
                    product_info[key] = value
                
                recommendations.append(product_info)
            except:
                continue
        
        if recommendations:
            recs_df = pd.DataFrame(recommendations)
            recs_df = recs_df.sort_values('ensemble_score', ascending=False)
            return recs_df.head(n_recommendations)
        else:
            return pd.DataFrame()
    
    def get_similar_products(self, product_idx, n_recommendations=10):
        """Get similar products using TF-IDF"""
        return self.tfidf_model.get_similar(product_idx, n_recommendations)
    
    def predict_rating(self, brand, category, reviews, wishlist, price=50000):
        """Predict rating for new product"""
        return self.rf_model.predict_rating(brand, category, reviews, wishlist, price)
    
    def load_trained_models(self):
        """Load all pre-trained models"""
        print("📥 Loading pre-trained models...")
        
        if not self.load_data():
            return False
        
        models_loaded = 0
        
        # Load individual models
        if self.tfidf_model.load_model():
            models_loaded += 1
            print("✅ TF-IDF model loaded")
        
        if self.svd_model.load_model():
            models_loaded += 1
            print("✅ SVD model loaded")
        
        if self.rf_model.load_model():
            models_loaded += 1
            print("✅ RF model loaded")
        
        # Load ensemble
        if self.load_system():
            print("✅ Ensemble system loaded")
        
        if models_loaded >= 2:
            self.is_trained = True
            print(f"✅ System ready! ({models_loaded}/3 models loaded)")
            return True
        else:
            print(f"❌ Insufficient models ({models_loaded}/3)")
            return False
    
    def save_system(self):
        """Save ensemble system"""
        try:
            with open('models/ensemble_system.pkl', 'wb') as f:
                pickle.dump({
                    'meta_learner': self.meta_learner,
                    'meta_scaler': self.meta_scaler,
                    'is_trained': self.is_trained
                }, f)
        except Exception as e:
            print(f"❌ Save error: {e}")
    
    def load_system(self):
        """Load ensemble system"""
        try:
            with open('models/ensemble_system.pkl', 'rb') as f:
                data = pickle.load(f)
            self.meta_learner = data['meta_learner']
            self.meta_scaler = data['meta_scaler']
            self.is_trained = data['is_trained']
            return True
        except:
            return False


# Backward compatibility aliases
HybridRecommenderSystem = EnsembleHybridSystem
CompactHybridRecommender = EnsembleHybridSystem


def main():
    """Test ensemble system"""
    print("🔀 TESTING ENSEMBLE SYSTEM")
    print("="*40)
    
    system = EnsembleHybridSystem()
    
    # Try loading first
    if system.load_trained_models():
        print("✅ Loaded existing models!")
    else:
        print("Training new models...")
        try:
            results = system.train_all_models()
            print(f"✅ Training completed: {results}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return
    
    # Test recommendations
    try:
        recs = system.get_recommendations(n_recommendations=3)
        print(f"\n📋 Recommendations:")
        for i, (_, p) in enumerate(recs.iterrows(), 1):
            print(f"{i}. {p['product_name']} - Ensemble: {p.get('ensemble_score', 0):.3f}")
    except Exception as e:
        print(f"❌ Testing failed: {e}")


if __name__ == "__main__":
    main()