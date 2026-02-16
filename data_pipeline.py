from preprocessing_data import SkincareDataCleaner
from feature_engineering import SkincareFeatureEngineer
import pandas as pd
import os

class SkincarePipeline:
    """Simplified data pipeline"""
    
    def __init__(self):
        self.output_dir = 'dataset/processed/'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_full_pipeline(self):
        """Run complete pipeline"""
        print("🚀 SKINCARE DATA PIPELINE")
        print("="*40)
        
        # Step 1: Clean data
        print("\n📋 Step 1: Data Cleaning")
        cleaner = SkincareDataCleaner()
        df_clean = cleaner.run()
        
        # Step 2: Feature engineering
        print("\n📋 Step 2: Feature Engineering")
        engineer = SkincareFeatureEngineer()
        df_processed = engineer.run()
        
        # Step 3: Final dataset (no sampling for full data)
        print(f"\n📋 Step 3: Final Dataset")
        df_final = df_processed.copy()
        
        # Only create sample if dataset is huge (>8000)
        if len(df_processed) > 8000:
            print(f"📋 Large dataset ({len(df_processed)}), creating sample...")
            df_sample = df_processed.sample(n=5000, random_state=42).reset_index(drop=True)
            
            sample_path = os.path.join(self.output_dir, 'skincare_sample.csv')
            df_sample.to_csv(sample_path, index=False)
            print(f"💾 Sample saved: {sample_path}")
        
        # Summary
        print("\n✅ PIPELINE COMPLETED!")
        print("="*40)
        print(f"📊 Final Results:")
        print(f"   Products: {len(df_final):,}")
        print(f"   Brands: {df_final['brand_name'].nunique()}")
        print(f"   Categories: {df_final['default_category'].nunique()}")
        print(f"   Avg Rating: {df_final['average_rating'].mean():.2f}")
        print(f"   Features: {df_final.shape[1]}")
        
        print(f"\n📁 Generated Files:")
        print(f"   ✅ dataset/processed/skincare_cleaned.csv")
        print(f"   ✅ dataset/processed/skincare_processed.csv")
        print(f"   ✅ dataset/processed/skincare_ml_features.csv")
        if len(df_processed) > 8000:
            print(f"   ✅ dataset/processed/skincare_sample.csv")
        
        return df_final
    
    def load_processed_data(self):
        """Load existing processed data"""
        # Try processed data first
        paths = [
            'dataset/processed/skincare_processed.csv',
            'dataset/processed/skincare_sample.csv'
        ]
        
        for path in paths:
            try:
                df = pd.read_csv(path)
                print(f"📥 Loaded existing: {path} ({len(df)} products)")
                return df
            except FileNotFoundError:
                continue
        
        # If no data exists, run pipeline
        print("❌ No processed data found. Running pipeline...")
        return self.run_full_pipeline()
    
    def check_data_quality(self, df):
        """Quick data quality check"""
        print(f"\n🔍 Data Quality Check:")
        print(f"   Missing values: {df.isnull().sum().sum()}")
        print(f"   Duplicate products: {df.duplicated().sum()}")
        print(f"   Rating range: {df['average_rating'].min():.1f} - {df['average_rating'].max():.1f}")
        print(f"   Review range: {df['total_reviews'].min()} - {df['total_reviews'].max():,}")
        
        # Check for issues
        issues = []
        if df['average_rating'].min() < 1 or df['average_rating'].max() > 5:
            issues.append("Invalid ratings")
        if df['total_reviews'].max() > 20000:
            issues.append("Extreme review counts")
        if df['brand_name'].nunique() < 10:
            issues.append("Too few brands")
        
        if issues:
            print(f"   ⚠️ Issues found: {', '.join(issues)}")
        else:
            print(f"   ✅ Data quality looks good!")

def main():
    """Main pipeline execution"""
    pipeline = SkincarePipeline()
    
    try:
        # Load or create processed data
        df = pipeline.load_processed_data()
        
        # Quality check
        pipeline.check_data_quality(df)
        
        # Show sample
        print(f"\n📋 Sample Data:")
        sample_cols = ['product_name', 'brand_name', 'default_category', 'average_rating']
        print(df[sample_cols].head(3).to_string(index=False))
        
        print(f"\n🎉 Pipeline ready! Dataset: {len(df):,} products")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        print("Please check your dataset file and try again.")

if __name__ == "__main__":
    main()