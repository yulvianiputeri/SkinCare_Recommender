import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDFContentModel:
    def __init__(self):
        # Optimized TF-IDF settings for better similarity
        self.tfidf = TfidfVectorizer(
            max_features=1000,  # Reduced for less sparsity
            ngram_range=(1,2), 
            stop_words='english',
            min_df=2,  # Word must appear in at least 2 documents
            max_df=0.8,  # Ignore words that appear in >80% of documents
            sublinear_tf=True,  # Use log scaling
            norm='l2'  # L2 normalization
        )
        self.content_matrix = None
        self.df = None
    
    def load_data(self):
        paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
        for path in paths:
            try:
                self.df = pd.read_csv(path)
                return True
            except:
                continue
        return False
    
    def prepare_content(self):
        """Enhanced content preparation for better similarity"""
        content_texts = []
        
        for _, row in self.df.iterrows():
            # Base content
            content_parts = []
            
            # Product name processing
            product_name = str(row['product_name']).lower()
            # Extract key terms
            key_terms = []
            skincare_keywords = ['serum', 'cream', 'day', 'night', 'brightening', 'whitening', 
                               'anti', 'aging', 'vitamin', 'essence', 'moisturizer', 'cleansing',
                               'toner', 'mask', 'oil', 'gel', 'lotion', 'sunscreen', 'spf']
            
            for keyword in skincare_keywords:
                if keyword in product_name:
                    key_terms.append(keyword)
            
            # Add repeated key terms for emphasis
            content_parts.extend(key_terms * 2)  # Emphasize important terms
            
            # Add product name
            content_parts.append(product_name)
            
            # Add brand (repeated for brand similarity)
            brand = str(row['brand_name']).lower()
            content_parts.extend([brand] * 2)
            
            # Add category (repeated for category similarity)  
            category = str(row['default_category']).lower()
            content_parts.extend([category] * 3)
            
            # Add price tier for similar price range
            price = row.get('price_numeric', 50000)
            if price < 100000:
                content_parts.extend(['affordable', 'budget'] * 2)
            elif price < 300000:
                content_parts.extend(['midrange', 'moderate'] * 2)
            else:
                content_parts.extend(['premium', 'luxury'] * 2)
            
            # Add rating tier for quality similarity
            rating = row.get('average_rating', 4.0)
            if rating >= 4.5:
                content_parts.extend(['excellent', 'top', 'best'] * 2)
            elif rating >= 4.0:
                content_parts.extend(['good', 'quality'] * 2)
            else:
                content_parts.append('average')
            
            # Combine all parts
            content_text = ' '.join(content_parts)
            content_texts.append(content_text)
        
        self.df['content_text'] = content_texts
    
    def train(self):
        print("🔤 Training TF-IDF model...")
        if not self.load_data():
            return False
        
        self.prepare_content()
        self.content_matrix = self.tfidf.fit_transform(self.df['content_text'])
        
        with open('models/tfidf_model.pkl', 'wb') as f:
            pickle.dump({
                'tfidf': self.tfidf,
                'content_matrix': self.content_matrix
            }, f)
        
        print(f"✅ TF-IDF trained - Matrix: {self.content_matrix.shape}")
        return True
    
    def load_model(self):
        try:
            if not self.load_data():
                return False
            with open('models/tfidf_model.pkl', 'rb') as f:
                data = pickle.load(f)
            self.tfidf = data['tfidf']
            self.content_matrix = data['content_matrix']
            return True
        except:
            return False
    
    def get_score(self, product_idx, user_preferences=None):
        base_score = self.df.iloc[product_idx]['average_rating'] / 5.0
        
        if user_preferences and self.content_matrix is not None:
            user_text = ""
            if user_preferences.get('categories'):
                # Repeat categories for emphasis
                user_text += " ".join(user_preferences['categories'] * 3) + " "
            if user_preferences.get('brands'):
                # Repeat brands for emphasis  
                user_text += " ".join(user_preferences['brands'] * 2) + " "
            
            if user_text.strip():
                try:
                    user_vector = self.tfidf.transform([user_text.lower()])
                    product_vector = self.content_matrix[product_idx]
                    similarity = cosine_similarity(user_vector, product_vector)[0][0]
                    
                    # Boost similarity score
                    boosted_similarity = min(similarity * 1.5, 1.0)  # 1.5x boost, cap at 1.0
                    return boosted_similarity * 0.7 + base_score * 0.3
                except:
                    pass
        
        return base_score + np.random.normal(0, 0.03)
    
    def get_similar(self, product_idx, n=10):
        if self.content_matrix is None:
            return pd.DataFrame()
        
        try:
            product_vector = self.content_matrix[product_idx]
            similarities = cosine_similarity(product_vector, self.content_matrix).flatten()
            
            # Boost similarities for better scores
            similarities = np.power(similarities, 0.7)  # Power boost to increase scores
            
            similar_indices = similarities.argsort()[::-1][1:n+1]
            
            result = self.df.iloc[similar_indices].copy()
            result['similarity_score'] = similarities[similar_indices]
            result['similarity_reason'] = 'Enhanced TF-IDF Content Similarity'
            
            return result
        except:
            return pd.DataFrame()