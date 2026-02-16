import os
import shutil
from pathlib import Path

class ProjectCleanup:
    """Clean up redundant files from project"""
    
    def __init__(self):
        self.archive_dir = 'archive'
        self.backup_dir = 'archive/backup_models'
        
        self.files_to_archive = [
            'hybrid_model.py',           
            'rf_model.py',               
            'apply_patch.py',            
            'quick_fix.py',              
            'baseline_comparison.py',   
            'visualization.py',       
            'run_all_revisions.py',     
            'README.md',                 
        ]
        
        # Essential files to KEEP
        self.essential_files = [
            'app.py',                    # Main Streamlit app
            'analytics.py',              # Dashboard
            'ensemble_system.py',        # Core system
            'tfidf_model.py',            # TF-IDF component
            'svd_model.py',              # SVD component
            'rf_model_improved.py',      # RF component (IMPROVED)
            'utils.py',                  # Utilities
            'requirements.txt',          # Dependencies
            'data_pipeline.py',          # Data pipeline
            'preprocessing_data.py',     # Preprocessing
            'feature_engineering.py',    # Feature engineering
        ]
    
    def create_directories(self):
        """Create archive directories"""
        print("\n📁 Creating archive directories...")
        os.makedirs(self.archive_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        print(f"   ✅ Created: {self.archive_dir}/")
        print(f"   ✅ Created: {self.backup_dir}/")
    
    def archive_file(self, filename):
        """Archive a single file"""
        if os.path.exists(filename):
            dest = os.path.join(self.archive_dir, filename)
            try:
                shutil.move(filename, dest)
                print(f"   ✅ Archived: {filename}")
                return True
            except Exception as e:
                print(f"   ⚠️ Could not archive {filename}: {e}")
                return False
        else:
            print(f"   ⏭️ Skip: {filename} (not found)")
            return False
    
    def backup_models(self):
        """Backup old model files"""
        print("\n🗄️ Backing up old models...")
        
        model_files = [
            'models/rf_model.pkl',
            'rf_model_old_backup.py',
            'rf_model_improved_BROKEN.py',
            'rf_model_original.py',
        ]
        
        backed_up = 0
        for model in model_files:
            if os.path.exists(model):
                dest = os.path.join(self.backup_dir, os.path.basename(model))
                try:
                    shutil.copy(model, dest)
                    print(f"   ✅ Backed up: {model}")
                    backed_up += 1
                except Exception as e:
                    print(f"   ⚠️ Could not backup {model}: {e}")
        
        if backed_up == 0:
            print("   ℹ️ No old models to backup")
    
    def archive_redundant_files(self):
        """Archive all redundant files"""
        print("\n📦 Archiving redundant files...")
        archived_count = 0
        
        for filename in self.files_to_archive:
            if self.archive_file(filename):
                archived_count += 1
        
        print(f"\n   📊 Total archived: {archived_count}/{len(self.files_to_archive)} files")
    
    def list_remaining_files(self):
        """List remaining essential files"""
        print("\n📋 Remaining Python files:")
        
        py_files = sorted([f for f in os.listdir('.') if f.endswith('.py')])
        
        for filename in py_files:
            if filename in self.essential_files:
                print(f"   ✅ {filename}")
            else:
                print(f"   ⚠️ {filename} (not in essential list)")
        
        print(f"\n   📊 Total: {len(py_files)} Python files")
    
    def create_cleanup_summary(self):
        """Create summary report"""
        summary_path = os.path.join(self.archive_dir, 'CLEANUP_SUMMARY.txt')
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PROJECT CLEANUP SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write("ARCHIVED FILES:\n")
            f.write("-"*70 + "\n")
            for filename in self.files_to_archive:
                status = "✅ Archived" if os.path.exists(os.path.join(self.archive_dir, filename)) else "⏭️ Not found"
                f.write(f"{status}: {filename}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("ESSENTIAL FILES (KEPT):\n")
            f.write("-"*70 + "\n")
            for filename in self.essential_files:
                status = "✅ Present" if os.path.exists(filename) else "❌ Missing"
                f.write(f"{status}: {filename}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("NOTES:\n")
            f.write("-"*70 + "\n")
            f.write("- Redundant files moved to archive/ folder\n")
            f.write("- Old models backed up to archive/backup_models/\n")
            f.write("- Essential files remain in project root\n")
            f.write("- You can restore files from archive/ if needed\n")
            f.write("\n")
        
        print(f"\n📄 Summary created: {summary_path}")
    
    def run_cleanup(self):
        """Run complete cleanup process"""
        print("="*70)
        print("🧹 PROJECT CLEANUP")
        print("="*70)
        
        print("\nThis script will:")
        print("  1. Create archive directories")
        print("  2. Backup old model files")
        print("  3. Archive redundant files")
        print("  4. List remaining essential files")
        print("  5. Create cleanup summary")
        
        response = input("\n⚠️ Continue? (y/n): ")
        
        if response.lower() != 'y':
            print("\n❌ Cleanup cancelled")
            return
        
        # Execute cleanup steps
        self.create_directories()
        self.backup_models()
        self.archive_redundant_files()
        self.list_remaining_files()
        self.create_cleanup_summary()
        
        # Final message
        print("\n" + "="*70)
        print("✅ CLEANUP COMPLETED!")
        print("="*70)
        
        print("\n📊 Summary:")
        print(f"   - Archived files: {self.archive_dir}/")
        print(f"   - Backed up models: {self.backup_dir}/")
        print(f"   - Summary report: {self.archive_dir}/CLEANUP_SUMMARY.txt")
        
        print("\n💡 What's next:")
        print("   1. Review remaining files")
        print("   2. Test your app: streamlit run app.py")
        print("   3. If everything works, you can delete archive/ folder")
        
        print("\n⚠️ To restore archived files:")
        print(f"   mv {self.archive_dir}/[filename] .")


def main():
    """Main cleanup execution"""
    cleanup = ProjectCleanup()
    cleanup.run_cleanup()


if __name__ == "__main__":
    main()