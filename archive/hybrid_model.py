# Simple Hybrid Model - Main Entry Point
# Just imports and re-exports the ensemble system

from ensemble_system import EnsembleHybridSystem

# Main aliases for backward compatibility
HybridRecommenderSystem = EnsembleHybridSystem
CompactHybridRecommender = EnsembleHybridSystem

# For direct imports
ContentBasedModel = EnsembleHybridSystem  # Will use tfidf_model internally
CollaborativeModel = EnsembleHybridSystem  # Will use svd_model internally  
RatingPredictionModel = EnsembleHybridSystem  # Will use rf_model internally

def main():
    """Test the hybrid system"""
    system = EnsembleHybridSystem()
    
    if system.load_trained_models():
        print("✅ Models loaded!")
    else:
        print("🚀 Training models...")
        system.train_all_models()
    
    # Quick test
    recs = system.get_recommendations(n_recommendations=3)
    print(f"📋 Got {len(recs)} recommendations")

if __name__ == "__main__":
    main()