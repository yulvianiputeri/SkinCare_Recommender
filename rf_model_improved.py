import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class RandomForestModel:
    def __init__(self):
        # Optimized parameters
        self.rf = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42, 
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.df = None
        self.feature_names = None
    
    def load_data(self):
        paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
        for path in paths:
            try:
                self.df = pd.read_csv(path)
                print(f"📥 RF loaded: {len(self.df):,} products from {path}")
                return True
            except:
                continue
        print("❌ RF: No data found")
        return False
    
    def prepare_features(self):
        """Prepare features - ONLY using columns that EXIST"""
        print("🔧 Preparing features...")
        
        # Basic encoding
        le_brand = LabelEncoder()
        le_category = LabelEncoder()
        
        self.df['brand_encoded'] = le_brand.fit_transform(self.df['brand_name'].astype(str))
        self.df['category_encoded'] = le_category.fit_transform(self.df['default_category'].astype(str))
        
        # Log transforms
        self.df['log_reviews'] = np.log1p(self.df['total_reviews'].fillna(0))
        self.df['log_wishlist'] = np.log1p(self.df['total_in_wishlist'].fillna(0))
        
        # Check and create additional features
        if 'log_price' not in self.df.columns:
            if 'price_numeric' in self.df.columns:
                self.df['log_price'] = np.log1p(self.df['price_numeric'].fillna(50000))
            else:
                self.df['log_price'] = 10.0  # default
        
        if 'popularity_score' not in self.df.columns:
            max_reviews = self.df['log_reviews'].max()
            max_wishlist = self.df['log_wishlist'].max()
            
            if max_reviews > 0 and max_wishlist > 0:
                self.df['popularity_score'] = (
                    self.df['average_rating'] * 0.4 + 
                    (self.df['log_reviews'] / max_reviews) * 0.3 + 
                    (self.df['log_wishlist'] / max_wishlist) * 0.3
                )
            else:
                self.df['popularity_score'] = self.df['average_rating'] * 0.8
        
        if 'review_wishlist_ratio' not in self.df.columns:
            self.df['review_wishlist_ratio'] = (
                self.df['total_in_wishlist'] / (self.df['total_reviews'] + 1)
            ).clip(0, 10)
        
        print("✅ Features prepared (7 core features)")
    
    def train(self):
        """Train Random Forest with ONLY existing features"""
        print("🌳 Training Random Forest (FIXED)...")
        if not self.load_data():
            return {'r2': 0.0, 'mae': 0.0, 'rmse': 0.0}
        
        self.prepare_features()
        
        # Use ONLY 7 features that exist
        self.feature_names = [
            'brand_encoded', 
            'category_encoded', 
            'log_reviews', 
            'log_wishlist',
            'log_price', 
            'popularity_score', 
            'review_wishlist_ratio'
        ]
        
        # Verify all features exist
        missing_features = [f for f in self.feature_names if f not in self.df.columns]
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
            return {'r2': 0.0, 'mae': 0.0, 'rmse': 0.0}
        
        X = self.df[self.feature_names]
        y = self.df['average_rating']
        
        # Remove any rows with NaN
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            print(f"❌ Too few samples: {len(X)}")
            return {'r2': 0.0, 'mae': 0.0, 'rmse': 0.0}
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train), 
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test), 
            columns=self.feature_names,
            index=X_test.index
        )
        
        # Train
        self.rf.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.rf.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        # Save model
        with open('models/rf_model.pkl', 'wb') as f:
            pickle.dump({
                'rf': self.rf,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }, f)
        
        print(f"✅ RF trained (FIXED)")
        print(f"   📊 R²: {r2:.3f}")
        print(f"   📊 MAE: {mae:.3f}")
        print(f"   📊 RMSE: {rmse:.3f}")
        
        if r2 < 0:
            print(f"   ⚠️ WARNING: Negative R² indicates poor model fit")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n📈 Feature Importance:")
        for idx, row in feature_importance.iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        return {
            'r2': r2, 
            'mae': mae, 
            'rmse': rmse, 
            'feature_importance': feature_importance
        }
    
    def load_model(self):
        try:
            if not self.load_data():
                return False
            
            with open('models/rf_model.pkl', 'rb') as f:
                data = pickle.load(f)
            
            self.rf = data['rf']
            self.scaler = data['scaler']
            self.feature_names = data.get('feature_names', [
                'brand_encoded', 'category_encoded', 'log_reviews', 
                'log_wishlist', 'log_price', 'popularity_score', 
                'review_wishlist_ratio'
            ])
            
            # Prepare features after loading
            self.prepare_features()
            
            return True
        except Exception as e:
            print(f"⚠️ Load model error: {e}")
            return False
    
    def get_score(self, product_idx):
        """Get prediction score for a product"""
        try:
            if self.df is None:
                return 0.5
            
            product = self.df.iloc[product_idx]
            
            if self.feature_names is None:
                self.feature_names = [
                    'brand_encoded', 'category_encoded', 'log_reviews', 
                    'log_wishlist', 'log_price', 'popularity_score', 
                    'review_wishlist_ratio'
                ]
            
            # Get feature values
            feature_values = []
            for feat in self.feature_names:
                if feat in product:
                    feature_values.append(product[feat])
                else:
                    # Fallback values
                    if feat in ['brand_encoded', 'category_encoded']:
                        feature_values.append(0)
                    else:
                        feature_values.append(1.0)
            
            features = pd.DataFrame([feature_values], columns=self.feature_names)
            features_scaled = self.scaler.transform(features)
            
            predicted = self.rf.predict(features_scaled)[0]
            
            # Normalize to 0-1
            score = (predicted / 5.0) + np.random.normal(0, 0.01)
            return np.clip(score, 0.1, 1.0)
            
        except Exception as e:
            print(f"Warning: get_score failed - {e}")
            return 0.5
    
    def predict_rating(self, brand, category, reviews, wishlist, price=50000):
        """Predict rating for a new product"""
        try:
            # Simple encoding
            brand_encoded = hash(brand) % 100
            category_encoded = hash(category) % 50
            
            log_reviews = np.log1p(reviews)
            log_wishlist = np.log1p(wishlist)
            log_price = np.log1p(price)
            popularity = 3.5  # default
            ratio = wishlist / (reviews + 1)
            
            features = pd.DataFrame([[
                brand_encoded, 
                category_encoded, 
                log_reviews, 
                log_wishlist,
                log_price,
                popularity,
                ratio
            ]], columns=self.feature_names)
            
            features_scaled = self.scaler.transform(features)
            prediction = self.rf.predict(features_scaled)[0]
            
            return np.clip(prediction, 1.0, 5.0)
        except Exception as e:
            print(f"Warning: predict_rating failed - {e}")
            return 3.5


def main():
    """Test fixed RF model"""
    print("🧪 TESTING FIXED RF MODEL")
    print("="*50)
    
    rf = RandomForestModel()
    results = rf.train()
    
    print(f"\n✅ Training complete!")
    print(f"📊 Final Results:")
    print(f"   R²: {results['r2']:.3f}")
    print(f"   MAE: {results['mae']:.3f}")
    print(f"   RMSE: {results['rmse']:.3f}")
    
    if results['r2'] > 0.3:
        print(f"\n🎉 SUCCESS! Model is working well")
    elif results['r2'] > 0:
        print(f"\n⚠️ Model works but R² is low. Consider more data/features")
    else:
        print(f"\n❌ Model has issues. R² should not be negative!")


if __name__ == "__main__":
    main()