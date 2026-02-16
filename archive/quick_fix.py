#!/usr/bin/env python3
# Quick Fix Script - Replace RF model with working version

import os
import shutil

def quick_fix():
    """Replace broken RF model with fixed version"""
    print("🔧 QUICK FIX - Replacing RF Model")
    print("="*50)
    
    # Backup broken version
    if os.path.exists('rf_model_improved.py'):
        print("📦 Backing up broken rf_model_improved.py...")
        shutil.copy('rf_model_improved.py', 'rf_model_improved_BROKEN.py')
        print("   ✅ Saved as rf_model_improved_BROKEN.py")
    
    # Backup original if exists
    if os.path.exists('rf_model.py'):
        print("\n📦 Backing up original rf_model.py...")
        shutil.copy('rf_model.py', 'rf_model_original.py')
        print("   ✅ Saved as rf_model_original.py")
    
    # Copy fixed version
    if os.path.exists('rf_model_fixed.py'):
        print("\n🔄 Installing fixed RF model...")
        shutil.copy('rf_model_fixed.py', 'rf_model.py')
        print("   ✅ rf_model.py now uses FIXED version (7 features)")
    else:
        print("\n❌ rf_model_fixed.py not found!")
        print("   Please download it from Claude first")
        return False
    
    print("\n✅ QUICK FIX COMPLETED!")
    print("\n📝 What changed:")
    print("   - Removed missing columns (total_recommended_count, total_repurchase_yes_count)")
    print("   - Now uses only 7 features that EXIST in your dataset")
    print("   - Better error handling")
    
    print("\n🚀 Next step:")
    print("   Run: python run_all_revisions.py")
    
    return True

if __name__ == "__main__":
    quick_fix()