import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

class SkincareFeatureEngineer:
    """Simplified feature engineering"""
    
    def __init__(self):
        self.input_dir = 'dataset/processed/'
        self.output_dir = 'dataset/processed/'
        self.scaler = StandardScaler()
    
    def load_cleaned_data(self):
        """Load cleaned data"""
        input_path = os.path.join(self.input_dir, 'skincare_cleaned.csv')
        try:
            df = pd.read_csv(input_path)
            print(f"📥 Loaded: {len(df)} cleaned products")
            return df
        except FileNotFoundError:
            raise FileNotFoundError("❌ Run preprocessing_data.py first!")
    
    def create_basic_features(self, df):
        """Create basic engineered features"""
        print("⚙️ Creating features...")
        
        # Safe numerical features
        df['log_reviews'] = np.log1p(df['total_reviews'].clip(0, 10000))
        df['log_wishlist'] = np.log1p(df['total_in_wishlist'].clip(0, 5000))
        df['log_price'] = np.log1p(df['price_numeric'].clip(1000, 5000000))
        
        # Ratio features
        df['wishlist_review_ratio'] = (
            df['total_in_wishlist'] / (df['total_reviews'] + 1)
        ).clip(0, 10)
        
        # Popularity score
        df['popularity_score'] = (
            df['average_rating'] * 0.4 + 
            (df['log_reviews'] / df['log_reviews'].max()) * 0.3 + 
            (df['log_wishlist'] / df['log_wishlist'].max()) * 0.3
        )
        
        print("✅ Basic features created")
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        print("🔤 Encoding categories...")
        
        # Limit to top categories for stability
        top_brands = df['brand_name'].value_counts().head(100).index
        top_categories = df['default_category'].value_counts().head(50).index
        
        df['brand_limited'] = df['brand_name'].apply(
            lambda x: x if x in top_brands else 'Other'
        )
        df['category_limited'] = df['default_category'].apply(
            lambda x: x if x in top_categories else 'Other'
        )
        
        # Label encoding
        brand_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        
        df['brand_encoded'] = brand_encoder.fit_transform(df['brand_limited'])
        df['category_encoded'] = category_encoder.fit_transform(df['category_limited'])
        
        print("✅ Categories encoded")
        return df
    
    def create_price_tiers(self, df):
        """Create price tier features"""
        print("💰 Creating price tiers...")
        
        # Calculate price percentiles
        price_33 = df['price_numeric'].quantile(0.33)
        price_67 = df['price_numeric'].quantile(0.67)
        
        def get_price_tier(price):
            if price <= price_33:
                return 'Budget'
            elif price <= price_67:
                return 'Mid-Range'
            else:
                return 'Premium'
        
        df['price_tier'] = df['price_numeric'].apply(get_price_tier)
        
        print("✅ Price tiers created")
        return df
    
    def scale_features(self, df):
        """Scale numerical features"""
        print("📏 Scaling features...")
        
        # Select features to scale
        features_to_scale = [
            'brand_encoded', 'category_encoded', 'log_reviews', 
            'log_wishlist', 'log_price', 'wishlist_review_ratio', 
            'popularity_score', 'average_rating'
        ]
        
        # Ensure all features exist
        for feature in features_to_scale:
            if feature not in df.columns:
                df[feature] = 0
        
        # Scale features
        try:
            feature_matrix = df[features_to_scale].fillna(0)
            scaled_matrix = self.scaler.fit_transform(feature_matrix)
            
            # Add scaled features
            for i, feature in enumerate(features_to_scale):
                df[f'{feature}_scaled'] = scaled_matrix[:, i]
            
            print("✅ Features scaled")
        except Exception as e:
            print(f"⚠️ Scaling warning: {e}")
            # Fallback: copy original features
            for feature in features_to_scale:
                df[f'{feature}_scaled'] = df[feature]
        
        return df
    
    def save_processed_data(self, df):
        """Save processed data"""
        # Full processed data
        full_path = os.path.join(self.output_dir, 'skincare_processed.csv')
        df.to_csv(full_path, index=False)
        
        # ML features only
        ml_columns = [col for col in df.columns if col.endswith('_scaled')] + [
            'product_name', 'brand_name', 'default_category', 'average_rating',
            'total_reviews', 'total_in_wishlist', 'price_numeric'
        ]
        
        # Add product ID if missing
        if 'product_id' not in df.columns:
            df['product_id'] = [f'PROD_{i:04d}' for i in range(len(df))]
            ml_columns.append('product_id')
        
        # Filter existing columns
        existing_columns = [col for col in ml_columns if col in df.columns]
        df_ml = df[existing_columns]
        
        ml_path = os.path.join(self.output_dir, 'skincare_ml_features.csv')
        df_ml.to_csv(ml_path, index=False)
        
        print(f"💾 Saved: {full_path}")
        print(f"💾 Saved: {ml_path}")
        
        return full_path, ml_path
    
    def run(self):
        """Main feature engineering process"""
        print("🚀 Starting feature engineering...")
        
        try:
            # Load data
            df = self.load_cleaned_data()
            
            # Create features step by step
            df = self.create_basic_features(df)
            df = self.encode_categorical_features(df)
            df = self.create_price_tiers(df)
            df = self.scale_features(df)
            
            # Save results
            full_path, ml_path = self.save_processed_data(df)
            
            # Summary
            print(f"\n📈 Feature Engineering Summary:")
            print(f"   Products: {len(df):,}")
            print(f"   Total Features: {df.shape[1]}")
            print(f"   ML Features: {len([c for c in df.columns if c.endswith('_scaled')])}")
            print(f"   Brands: {df['brand_name'].nunique()}")
            print(f"   Categories: {df['default_category'].nunique()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {e}")
            
            # Create minimal fallback
            df = self.load_cleaned_data()
            
            # Add minimal required features
            df['brand_encoded'] = pd.factorize(df['brand_name'])[0]
            df['category_encoded'] = pd.factorize(df['default_category'])[0]
            df['log_reviews'] = np.log1p(df['total_reviews'])
            df['log_wishlist'] = np.log1p(df['total_in_wishlist'])
            
            # Add scaled versions (just copy for simplicity)
            for col in ['brand_encoded', 'category_encoded', 'log_reviews', 'log_wishlist']:
                df[f'{col}_scaled'] = df[col]
            
            self.save_processed_data(df)
            print("✅ Fallback processing completed")
            return df

def main():
    """Test feature engineering"""
    engineer = SkincareFeatureEngineer()
    df = engineer.run()
    print("🎉 Feature engineering completed!")

if __name__ == "__main__":
    main()