# Baseline Comparison Script
# Membandingkan: TF-IDF vs SVD vs RF vs Ensemble
# Metrik: R², MAE, RMSE, Precision@K, Recall@K

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import models
from tfidf_model import TFIDFContentModel
from svd_model import SVDCollaborativeModel
from rf_model_improved import RandomForestModel
from ensemble_system import EnsembleHybridSystem

class BaselineComparison:
    """Compare all models: TF-IDF, SVD, RF, Ensemble"""
    
    def __init__(self):
        self.df = None
        self.results = {}
        
    def load_data(self):
        """Load dataset"""
        paths = ['dataset/processed/skincare_processed.csv', 'dataset/processed/skincare_sample.csv']
        for path in paths:
            try:
                self.df = pd.read_csv(path)
                print(f"📥 Loaded: {len(self.df):,} products from {path}")
                return True
            except:
                continue
        return False
    
    def evaluate_single_model(self, model_name, predictions, actual):
        """Calculate metrics for a single model"""
        # R² Score
        r2 = r2_score(actual, predictions)
        
        # MAE (Mean Absolute Error)
        mae = mean_absolute_error(actual, predictions)
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        return {
            'model': model_name,
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
    
    def calculate_precision_recall_at_k(self, recommendations, ground_truth, k=10):
        """Calculate Precision@K and Recall@K"""
        # Get top K recommendations
        top_k_recs = set(recommendations[:k])
        relevant = set(ground_truth)
        
        # Precision@K
        precision = len(top_k_recs & relevant) / k if k > 0 else 0
        
        # Recall@K
        recall = len(top_k_recs & relevant) / len(relevant) if len(relevant) > 0 else 0
        
        return precision, recall
    
    def evaluate_tfidf_only(self, test_indices, y_test):
        """Evaluate TF-IDF content-based model only"""
        print("\n📋 Evaluating TF-IDF Only...")
        
        tfidf = TFIDFContentModel()
        if not tfidf.load_model():
            tfidf.train()
        
        predictions = []
        for idx in test_indices:
            score = tfidf.get_score(idx)
            # Convert score (0-1) to rating (1-5)
            pred_rating = score * 5.0
            predictions.append(pred_rating)
        
        result = self.evaluate_single_model('TF-IDF Only', predictions, y_test)
        print(f"   ✅ R²: {result['r2']:.3f}, MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
        return result
    
    def evaluate_svd_only(self, test_indices, y_test):
        """Evaluate SVD collaborative filtering only"""
        print("\n📋 Evaluating SVD Only...")
        
        svd = SVDCollaborativeModel()
        if not svd.load_model():
            svd.train()
        
        predictions = []
        for idx in test_indices:
            score = svd.get_score(idx)
            # Convert score (0-1) to rating (1-5)
            pred_rating = score * 5.0
            predictions.append(pred_rating)
        
        result = self.evaluate_single_model('SVD Only', predictions, y_test)
        print(f"   ✅ R²: {result['r2']:.3f}, MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
        return result
    
    def evaluate_rf_only(self, test_indices, y_test):
        """Evaluate Random Forest only"""
        print("\n📋 Evaluating RF Only...")
        
        rf = RandomForestModel()
        if not rf.load_model():
            rf_results = rf.train()
            return {
                'model': 'RF Only',
                'r2': rf_results['r2'],
                'mae': rf_results['mae'],
                'rmse': rf_results['rmse']
            }
        
        predictions = []
        for idx in test_indices:
            score = rf.get_score(idx)
            # Convert score (0-1) to rating (1-5)
            pred_rating = score * 5.0
            predictions.append(pred_rating)
        
        result = self.evaluate_single_model('RF Only', predictions, y_test)
        print(f"   ✅ R²: {result['r2']:.3f}, MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
        return result
    
    def evaluate_ensemble(self, test_indices, y_test):
        """Evaluate Ensemble system"""
        print("\n📋 Evaluating Ensemble...")
        
        ensemble = EnsembleHybridSystem()
        if not ensemble.load_trained_models():
            print("   Training ensemble from scratch...")
            ensemble.train_all_models()
        
        predictions = []
        for idx in test_indices:
            scores = ensemble.get_ensemble_score(idx)
            # Convert ensemble score (0-1) to rating (1-5)
            pred_rating = scores['ensemble_score'] * 5.0
            predictions.append(pred_rating)
        
        result = self.evaluate_single_model('Hybrid Ensemble', predictions, y_test)
        print(f"   ✅ R²: {result['r2']:.3f}, MAE: {result['mae']:.3f}, RMSE: {result['rmse']:.3f}")
        return result
    
    def run_comparison(self):
        """Run complete comparison"""
        print("🚀 BASELINE COMPARISON")
        print("="*60)
        
        if not self.load_data():
            print("❌ Cannot load data!")
            return None
        
        # Prepare test set
        print("\n📊 Preparing test set...")
        test_size = min(500, len(self.df) // 5)
        test_indices = np.random.choice(len(self.df), test_size, replace=False)
        y_test = self.df.iloc[test_indices]['average_rating'].values
        
        print(f"   Test size: {test_size} products")
        
        # Evaluate each model
        results = []
        
        # 1. TF-IDF Only
        try:
            tfidf_result = self.evaluate_tfidf_only(test_indices, y_test)
            results.append(tfidf_result)
        except Exception as e:
            print(f"   ⚠️ TF-IDF failed: {e}")
        
        # 2. SVD Only
        try:
            svd_result = self.evaluate_svd_only(test_indices, y_test)
            results.append(svd_result)
        except Exception as e:
            print(f"   ⚠️ SVD failed: {e}")
        
        # 3. RF Only
        try:
            rf_result = self.evaluate_rf_only(test_indices, y_test)
            results.append(rf_result)
        except Exception as e:
            print(f"   ⚠️ RF failed: {e}")
        
        # 4. Ensemble
        try:
            ensemble_result = self.evaluate_ensemble(test_indices, y_test)
            results.append(ensemble_result)
        except Exception as e:
            print(f"   ⚠️ Ensemble failed: {e}")
        
        self.results = pd.DataFrame(results)
        return self.results
    
    def print_results_table(self):
        """Print formatted results table"""
        if self.results is None or len(self.results) == 0:
            print("No results to display")
            return
        
        print("\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        print(self.results.to_string(index=False))
        print("="*60)
        
        # Find best model
        best_r2_idx = self.results['r2'].idxmax()
        best_model = self.results.iloc[best_r2_idx]
        
        print(f"\n🏆 BEST MODEL: {best_model['model']}")
        print(f"   R²: {best_model['r2']:.3f}")
        print(f"   MAE: {best_model['mae']:.3f}")
        print(f"   RMSE: {best_model['rmse']:.3f}")
        
        # Improvement
        if 'Hybrid Ensemble' in self.results['model'].values:
            ensemble_r2 = self.results[self.results['model'] == 'Hybrid Ensemble']['r2'].values[0]
            
            if 'TF-IDF Only' in self.results['model'].values:
                tfidf_r2 = self.results[self.results['model'] == 'TF-IDF Only']['r2'].values[0]
                improvement = ((ensemble_r2 - tfidf_r2) / tfidf_r2) * 100 if tfidf_r2 > 0 else 0
                print(f"\n📈 Ensemble improves TF-IDF by: {improvement:.1f}%")
            
            if 'SVD Only' in self.results['model'].values:
                svd_r2 = self.results[self.results['model'] == 'SVD Only']['r2'].values[0]
                improvement = ((ensemble_r2 - svd_r2) / svd_r2) * 100 if svd_r2 > 0 else 0
                print(f"📈 Ensemble improves SVD by: {improvement:.1f}%")
            
            if 'RF Only' in self.results['model'].values:
                rf_r2 = self.results[self.results['model'] == 'RF Only']['r2'].values[0]
                improvement = ((ensemble_r2 - rf_r2) / rf_r2) * 100 if rf_r2 > 0 else 0
                print(f"📈 Ensemble improves RF by: {improvement:.1f}%")
    
    def save_results(self, filename='comparison_results.csv'):
        """Save results to CSV"""
        if self.results is not None and len(self.results) > 0:
            self.results.to_csv(filename, index=False)
            print(f"\n💾 Results saved to: {filename}")
    
    def visualize_results(self, save_path='comparison_chart.png'):
        """Create visualization of comparison results"""
        if self.results is None or len(self.results) == 0:
            print("No results to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² Score
        axes[0].bar(self.results['model'], self.results['r2'], color='steelblue')
        axes[0].set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('R² Score')
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # MAE
        axes[1].bar(self.results['model'], self.results['mae'], color='coral')
        axes[1].set_title('MAE (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Mean Absolute Error')
        axes[1].grid(axis='y', alpha=0.3)
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # RMSE
        axes[2].bar(self.results['model'], self.results['rmse'], color='mediumseagreen')
        axes[2].set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Root Mean Squared Error')
        axes[2].grid(axis='y', alpha=0.3)
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Visualization saved to: {save_path}")
        plt.close()


def main():
    """Run baseline comparison"""
    comparison = BaselineComparison()
    
    # Run comparison
    results = comparison.run_comparison()
    
    if results is not None:
        # Print results
        comparison.print_results_table()
        
        # Save results
        comparison.save_results('baseline_comparison_results.csv')
        
        # Create visualization
        comparison.visualize_results('baseline_comparison_chart.png')
        
        print("\n✅ COMPARISON COMPLETED!")
        print("📄 Files generated:")
        print("   - baseline_comparison_results.csv")
        print("   - baseline_comparison_chart.png")
    else:
        print("\n❌ Comparison failed!")


if __name__ == "__main__":
    main()