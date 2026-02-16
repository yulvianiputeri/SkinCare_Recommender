# Patch Instructions
# Manual changes needed for using improved RF model

"""
IMPORTANT: Update these imports manually

1. In ensemble_system.py (line 18):
   
   CHANGE FROM:
   from rf_model import RandomForestModel
   
   CHANGE TO:
   try:
       from rf_model_improved import RandomForestModel
   except ImportError:
       from rf_model import RandomForestModel

2. In app.py (if it imports rf_model):
   
   Same change as above

3. Alternatively: Just rename files
   
   mv rf_model.py rf_model_old.py
   mv rf_model_improved.py rf_model.py
   
   This way all imports automatically use the improved version!
"""

import os
import shutil

def apply_patch():
    """Apply patch by renaming files"""
    print("🔧 APPLYING PATCH")
    print("="*50)
    
    # Backup old RF
    if os.path.exists('rf_model.py'):
        print("📦 Backing up old rf_model.py...")
        shutil.copy('rf_model.py', 'rf_model_old_backup.py')
        print("   ✅ Saved as rf_model_old_backup.py")
    
    # Replace with improved version
    if os.path.exists('rf_model_improved.py'):
        print("\n🔄 Replacing rf_model.py with improved version...")
        shutil.copy('rf_model_improved.py', 'rf_model.py')
        print("   ✅ rf_model.py now uses 9 features!")
    
    print("\n✅ PATCH APPLIED!")
    print("   Old version backed up to: rf_model_old_backup.py")
    print("   All imports will now use improved RF automatically")

if __name__ == "__main__":
    apply_patch()