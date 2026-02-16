import pandas as pd
import numpy as np
import re

def get_recommendation_explanation(product):
    """Generate simple explanations for recommendations"""
    explanations = []
    
    # Rating-based explanations
    rating = product.get('average_rating', 0)
    if rating >= 4.5:
        explanations.append("⭐ Exceptional quality")
    elif rating >= 4.0:
        explanations.append("⭐ High quality")
    elif rating >= 3.5:
        explanations.append("👍 Good quality")
    
    # Review-based explanations
    reviews = product.get('total_reviews', 0)
    if reviews >= 200:
        explanations.append("🔥 Bestseller")
    elif reviews >= 100:
        explanations.append("💬 Well-reviewed")
    elif reviews >= 50:
        explanations.append("📝 Popular")
    
    # Wishlist-based explanations
    wishlist = product.get('total_in_wishlist', 0)
    if wishlist >= 100:
        explanations.append("❤️ Highly desired")
    elif wishlist >= 50:
        explanations.append("💕 Popular choice")
    
    # Value explanations
    price = product.get('price_numeric', 0)
    if price > 0:
        if price < 100000 and rating >= 4.0:
            explanations.append("💰 Great value")
        elif price > 300000 and rating >= 4.5:
            explanations.append("💎 Premium choice")
    
    return explanations[:3]  # Return max 3 explanations

def format_price(price):
    """Format price with currency"""
    if pd.isna(price) or price <= 0:
        return "Price not available"
    return f"Rp{price:,.0f}"

def format_number(number):
    """Format large numbers with commas"""
    if pd.isna(number):
        return "0"
    return f"{int(number):,}"

def extract_product_keywords(product_name):
    """Extract keywords from product name"""
    if pd.isna(product_name):
        return []
    
    # Common skincare keywords
    keywords = []
    product_lower = product_name.lower()
    
    skincare_terms = [
        'serum', 'cream', 'cleanser', 'toner', 'moisturizer', 'sunscreen',
        'oil', 'mask', 'gel', 'lotion', 'essence', 'ampoule', 'balm'
    ]
    
    for term in skincare_terms:
        if term in product_lower:
            keywords.append(term.title())
    
    # Extract size if present
    size_match = re.search(r'(\d+)\s*(ml|g|oz)', product_lower)
    if size_match:
        keywords.append(f"{size_match.group(1)}{size_match.group(2)}")
    
    return keywords[:3]  # Return max 3 keywords

def categorize_product_type(category, product_name):
    """Simple product type categorization"""
    category_lower = category.lower() if category else ""
    product_lower = product_name.lower() if product_name else ""
    
    if any(term in category_lower or term in product_lower for term in ['serum', 'essence']):
        return "Treatment"
    elif any(term in category_lower or term in product_lower for term in ['cleanser', 'wash', 'foam']):
        return "Cleansing"
    elif any(term in category_lower or term in product_lower for term in ['moisturizer', 'cream', 'lotion']):
        return "Moisturizing"
    elif any(term in category_lower or term in product_lower for term in ['sunscreen', 'spf']):
        return "Protection"
    elif any(term in category_lower or term in product_lower for term in ['toner', 'mist']):
        return "Preparation"
    else:
        return "Other"

def calculate_value_score(price, rating, reviews):
    """Calculate a simple value score"""
    try:
        if price <= 0 or rating <= 0:
            return 0.5
        
        # Normalize price (lower is better for value)
        price_score = max(0.1, 1 - (price / 500000))  # Assuming max reasonable price 500k
        
        # Normalize rating
        rating_score = rating / 5.0
        
        # Normalize reviews (more reviews = more trustworthy)
        review_score = min(1.0, reviews / 200)  # Cap at 200 reviews for max score
        
        # Weighted combination
        value_score = (price_score * 0.4 + rating_score * 0.4 + review_score * 0.2)
        
        return round(value_score, 3)
    except:
        return 0.5

def get_product_summary(product):
    """Get a one-line product summary"""
    try:
        name = product.get('product_name', 'Unknown Product')[:30]
        brand = product.get('brand_name', 'Unknown')
        rating = product.get('average_rating', 0)
        reviews = product.get('total_reviews', 0)
        
        if len(name) > 30:
            name += "..."
        
        return f"{name} by {brand} - ⭐{rating:.1f} ({reviews:,} reviews)"
    except:
        return "Product information unavailable"

def filter_products_by_criteria(df, criteria):
    """Filter products based on user criteria"""
    filtered_df = df.copy()
    
    try:
        # Filter by brands
        if criteria.get('brands'):
            filtered_df = filtered_df[filtered_df['brand_name'].isin(criteria['brands'])]
        
        # Filter by categories
        if criteria.get('categories'):
            filtered_df = filtered_df[filtered_df['default_category'].isin(criteria['categories'])]
        
        # Filter by minimum rating
        if criteria.get('min_rating'):
            filtered_df = filtered_df[filtered_df['average_rating'] >= criteria['min_rating']]
        
        # Filter by price range
        if criteria.get('max_price'):
            filtered_df = filtered_df[filtered_df['price_numeric'] <= criteria['max_price']]
        
        if criteria.get('min_price'):
            filtered_df = filtered_df[filtered_df['price_numeric'] >= criteria['min_price']]
        
        # Filter by minimum reviews
        if criteria.get('min_reviews'):
            filtered_df = filtered_df[filtered_df['total_reviews'] >= criteria['min_reviews']]
        
        return filtered_df
    except Exception as e:
        print(f"Warning: Filter error - {e}")
        return df

# CSS styles for Streamlit
SIMPLE_CSS = """
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem; border-radius: 10px; color: white; text-align: center;
    margin: 0.5rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.product-card {
    background: #f8f9fa; padding: 1rem; border-radius: 8px; 
    border-left: 4px solid #667eea; margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.similar-card {
    background: #e8f4fd; padding: 1rem; border-radius: 8px; 
    border-left: 4px solid #2196f3; margin: 0.5rem 0;
}

.warning-card {
    background: #fff3cd; padding: 1rem; border-radius: 8px;
    border-left: 4px solid #ffc107; margin: 0.5rem 0;
}
</style>
"""

def main():
    """Test utilities"""
    # Test data
    test_product = {
        'product_name': 'NIVEA Luminous 630 Spotclear Intensive Treatment Serum 30mL',
        'brand_name': 'Nivea',
        'average_rating': 4.58,
        'total_reviews': 250,
        'total_in_wishlist': 120,
        'price_numeric': 150000
    }
    
    print("🧪 Testing utilities...")
    print(f"Explanations: {get_recommendation_explanation(test_product)}")
    print(f"Price: {format_price(test_product['price_numeric'])}")
    print(f"Keywords: {extract_product_keywords(test_product['product_name'])}")
    print(f"Value Score: {calculate_value_score(test_product['price_numeric'], test_product['average_rating'], test_product['total_reviews'])}")
    print(f"Summary: {get_product_summary(test_product)}")
    print("✅ Utilities test completed!")

if __name__ == "__main__":
    main()