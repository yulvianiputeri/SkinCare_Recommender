import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp

class SVDCollaborativeModel:
    def __init__(self):
        self.svd = TruncatedSVD(n_components=30, random_state=42)
        self.user_factors = None
        self.item_factors = None
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
    
    def create_interactions(self):
        interactions = []
        n_users = min(500, len(self.df) // 10)
        
        for user_id in range(n_users):
            n_ratings = np.random.randint(10, 30)
            products = np.random.choice(len(self.df), n_ratings, replace=False)
            
            for prod_idx in products:
                base_rating = self.df.iloc[prod_idx]['average_rating']
                rating = np.clip(base_rating + np.random.normal(0, 0.3), 1, 5)
                interactions.append({
                    'user_id': user_id,
                    'item_id': prod_idx,
                    'rating': rating
                })
        
        return pd.DataFrame(interactions)
    
    def train(self):
        print("🤖 Training SVD model...")
        if not self.load_data():
            return False
        
        # Create synthetic interactions
        interactions = self.create_interactions()
        
        # Build matrix
        user_item_matrix = interactions.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        # Apply SVD
        sparse_matrix = sp.csr_matrix(user_item_matrix.values)
        self.user_factors = self.svd.fit_transform(sparse_matrix)
        self.item_factors = self.svd.components_.T
        
        # Save model
        with open('models/svd_model.pkl', 'wb') as f:
            pickle.dump({
                'svd': self.svd,
                'user_factors': self.user_factors,
                'item_factors': self.item_factors
            }, f)
        
        print(f"✅ SVD trained - Users: {self.user_factors.shape[0]}, Items: {self.item_factors.shape[0]}")
        return True
    
    def load_model(self):
        try:
            if not self.load_data():
                return False
            with open('models/svd_model.pkl', 'rb') as f:
                data = pickle.load(f)
            self.svd = data['svd']
            self.user_factors = data['user_factors']
            self.item_factors = data['item_factors']
            return True
        except:
            return False
    
    def get_score(self, product_idx, user_id=0):
        try:
            if (self.user_factors is not None and 
                user_id < self.user_factors.shape[0] and 
                product_idx < self.item_factors.shape[0]):
                
                # Average multiple users
                user_scores = []
                for uid in range(min(3, self.user_factors.shape[0])):
                    predicted = np.dot(self.user_factors[uid], self.item_factors[product_idx])
                    user_scores.append(predicted + 3.5)
                
                avg_prediction = np.mean(user_scores)
                return np.clip(avg_prediction / 5.0, 0.2, 1.0)
            else:
                # Fallback
                product = self.df.iloc[product_idx]
                return (product['average_rating'] / 5.0) * 0.8
        except:
            return 0.5