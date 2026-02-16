import pandas as pd
import numpy as np
import re
import os

class SkincareDataCleaner:
    """Simplified data cleaning"""
    
    def __init__(self):
        self.output_dir = 'dataset/processed/'
        os.makedirs(self.output_dir, exist_ok=True)
    
    def clean_brand_name(self, brand_str):
        """Clean brand names"""
        if pd.isna(brand_str):
            return "Unknown"
        
        brand_str = str(brand_str)
        
        # Remove numeric prefix
        brand_str = re.sub(r'^\d+_', '', brand_str)
        
        # Take meaningful parts
        parts = brand_str.split('_')
        clean_parts = []
        
        for part in parts:
            if len(part) > 2 and not part.isdigit():
                clean_part = re.sub(r'[^a-zA-Z\s]', '', part)
                if len(clean_part) > 2:
                    clean_parts.append(clean_part)
        
        if clean_parts:
            result = clean_parts[0].capitalize()
            return result[:20]
        
        return "Unknown"
    
    def clean_category_name(self, category_str):
        """Clean category names"""
        if pd.isna(category_str):
            return "Other"
        
        category_str = str(category_str)
        
        # Remove unwanted characters
        category_str = re.sub(r'[{}]', '', category_str)
        category_str = re.sub(r'[^\w\s]', '', category_str)
        category_str = category_str.strip()
        
        # Simple mappings
        if 'face wash' in category_str.lower():
            return 'Face Wash'
        elif 'face cream' in category_str.lower():
            return 'Face Cream'
        elif 'serum' in category_str.lower():
            return 'Face Serum'
        elif 'body' in category_str.lower():
            return 'Body Care'
        elif len(category_str) > 0:
            return category_str.title()[:30]
        
        return "Other"
    
    def extract_price(self, price_str):
        """Extract price from string"""
        if pd.isna(price_str):
            return 50000
        
        numbers = re.findall(r'[\d,]+', str(price_str))
        if numbers:
            try:
                price = int(numbers[0].replace(',', ''))
                return max(1000, min(price, 5000000))  # Reasonable bounds
            except:
                pass
        return 50000
    
    def load_raw_data(self):
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
                    print(f"✅ Found dataset: {path} ({len(df)} rows)")
                    return df
            except FileNotFoundError:
                continue
        
        raise FileNotFoundError("❌ No dataset found!")
    
    def clean_data(self, df):
        """Main cleaning process"""
        print(f"🧹 Cleaning {len(df)} rows...")
        
        # Drop rows with missing essential data
        essential_cols = ['product_name', 'brand_name', 'average_rating']
        df = df.dropna(subset=essential_cols)
        
        # Clean text columns
        df['brand_name'] = df['brand_name'].apply(self.clean_brand_name)
        df['default_category'] = df['default_category'].apply(self.clean_category_name)
        
        # Clean numerical columns
        df['total_reviews'] = pd.to_numeric(df['total_reviews'], errors='coerce').fillna(0)
        df['total_in_wishlist'] = pd.to_numeric(df['total_in_wishlist'], errors='coerce').fillna(0)
        df['average_rating'] = pd.to_numeric(df['average_rating'], errors='coerce')
        
        # Extract price
        if 'price_range' in df.columns:
            df['price_numeric'] = df['price_range'].apply(self.extract_price)
        else:
            df['price_numeric'] = 50000  # Default price
        
        # Filter valid data
        df = df[(df['average_rating'] >= 1) & (df['average_rating'] <= 5)]
        df = df[df['total_reviews'] <= 10000]  # Remove outliers
        
        # Remove brands/categories with too few products
        brand_counts = df['brand_name'].value_counts()
        category_counts = df['default_category'].value_counts()
        
        valid_brands = brand_counts[brand_counts >= 3].index
        valid_categories = category_counts[category_counts >= 5].index
        
        df = df[df['brand_name'].isin(valid_brands)]
        df = df[df['default_category'].isin(valid_categories)]
        
        print(f"✅ Cleaned to {len(df)} products, {df['brand_name'].nunique()} brands")
        return df
    
    def save_cleaned_data(self, df):
        """Save cleaned data"""
        output_path = os.path.join(self.output_dir, 'skincare_cleaned.csv')
        df.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        return output_path
    
    def run(self):
        """Main cleaning process"""
        print("🚀 Starting data cleaning...")
        
        # Load and clean
        df_raw = self.load_raw_data()
        df_clean = self.clean_data(df_raw)
        
        # Save
        output_path = self.save_cleaned_data(df_clean)
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"   Raw → Clean: {len(df_raw):,} → {len(df_clean):,}")
        print(f"   Brands: {df_clean['brand_name'].nunique()}")
        print(f"   Categories: {df_clean['default_category'].nunique()}")
        print(f"   Avg Rating: {df_clean['average_rating'].mean():.2f}")
        
        return df_clean

def main():
    """Test preprocessing"""
    cleaner = SkincareDataCleaner()
    df = cleaner.run()
    print("🎉 Preprocessing completed!")

if __name__ == "__main__":
    main()