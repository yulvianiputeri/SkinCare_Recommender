# Visualization Generator for Paper
# Generate all charts needed for journal revision

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PaperVisualization:
    """Generate visualizations for paper revision"""
    
    def __init__(self):
        self.results_df = None
        
    def load_results(self, filepath='baseline_comparison_results.csv'):
        """Load comparison results"""
        try:
            self.results_df = pd.read_csv(filepath)
            print(f"✅ Loaded results: {len(self.results_df)} models")
            return True
        except:
            print("❌ Cannot load results. Run baseline_comparison.py first!")
            return False
    
    def create_main_comparison_chart(self, save_path='figures/main_comparison.png'):
        """Create main comparison chart with all metrics"""
        if self.results_df is None:
            print("No results loaded!")
            return
        
        fig = plt.figure(figsize=(16, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Color scheme
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        # 1. R² Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(self.results_df['model'], self.results_df['r2'], 
                        color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_title('(a) R² Score Comparison', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('R² Score', fontsize=12)
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Baseline (0.5)')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. MAE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(self.results_df['model'], self.results_df['mae'], 
                        color=colors, edgecolor='black', linewidth=1.5)
        ax2.set_title('(b) MAE Comparison', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Mean Absolute Error', fontsize=12)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. RMSE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        bars3 = ax3.bar(self.results_df['model'], self.results_df['rmse'], 
                        color=colors, edgecolor='black', linewidth=1.5)
        ax3.set_title('(c) RMSE Comparison', fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylabel('Root Mean Squared Error', fontsize=12)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Main comparison saved: {save_path}")
        plt.close()
    
    def create_improvement_chart(self, save_path='figures/improvement_chart.png'):
        """Create chart showing improvement percentages"""
        if self.results_df is None:
            print("No results loaded!")
            return
        
        # Get ensemble results
        ensemble_idx = self.results_df[self.results_df['model'] == 'Hybrid Ensemble'].index
        if len(ensemble_idx) == 0:
            print("No ensemble results found!")
            return
        
        ensemble = self.results_df.iloc[ensemble_idx[0]]
        
        # Calculate improvements
        improvements = []
        for idx, row in self.results_df.iterrows():
            if row['model'] != 'Hybrid Ensemble':
                improvement = ((ensemble['r2'] - row['r2']) / row['r2']) * 100 if row['r2'] > 0 else 0
                improvements.append({
                    'model': f"vs {row['model']}",
                    'improvement': improvement
                })
        
        if not improvements:
            return
        
        imp_df = pd.DataFrame(improvements)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(imp_df['model'], imp_df['improvement'], 
                       color='#27ae60', edgecolor='black', linewidth=1.5)
        
        ax.set_title('Hybrid Ensemble Improvement over Individual Models', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Improvement (%)', fontsize=12)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.axvline(x=0, color='black', linewidth=1)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.1f}%',
                   ha='left', va='center', fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Improvement chart saved: {save_path}")
        plt.close()
    
    def create_radar_chart(self, save_path='figures/radar_comparison.png'):
        """Create radar chart for multi-metric comparison"""
        if self.results_df is None:
            print("No results loaded!")
            return
        
        from math import pi
        
        # Prepare data - normalize metrics
        categories = ['R² Score', 'MAE\n(inverted)', 'RMSE\n(inverted)']
        N = len(categories)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        colors_radar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        
        for idx, row in self.results_df.iterrows():
            # Normalize values (higher is better for all)
            r2_norm = row['r2']
            mae_norm = 1 - (row['mae'] / self.results_df['mae'].max())  # Invert
            rmse_norm = 1 - (row['rmse'] / self.results_df['rmse'].max())  # Invert
            
            values = [r2_norm, mae_norm, rmse_norm]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], 
                   color=colors_radar[idx % len(colors_radar)])
            ax.fill(angles, values, alpha=0.15, color=colors_radar[idx % len(colors_radar)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Performance Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Radar chart saved: {save_path}")
        plt.close()
    
    def generate_all_figures(self):
        """Generate all figures for paper"""
        import os
        os.makedirs('figures', exist_ok=True)
        
        print("🎨 GENERATING FIGURES FOR PAPER")
        print("="*50)
        
        # Main comparison
        self.create_main_comparison_chart('figures/figure_1_main_comparison.png')
        
        # Improvement chart
        self.create_improvement_chart('figures/figure_2_improvement.png')
        
        # Radar chart
        self.create_radar_chart('figures/figure_3_radar_comparison.png')
        
        print("\n✅ ALL FIGURES GENERATED!")
        print("📁 Check 'figures/' folder for:")
        print("   - figure_1_main_comparison.png")
        print("   - figure_2_improvement.png")
        print("   - figure_3_radar_comparison.png")


def main():
    """Generate all visualizations"""
    viz = PaperVisualization()
    
    if viz.load_results():
        viz.generate_all_figures()
    else:
        print("\n⚠️ Run baseline_comparison.py first to generate results!")


if __name__ == "__main__":
    main()