"""Tab 4: Model Comparison"""
import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tfidf_model import TFIDFContentModel
from rf_model_improved import RandomForestModel


def render_comparison(df, hybrid_system):
    """Render comparison tab"""
    st.header("📊 Model Performance Comparison")
    
    st.markdown("""
    This tab compares the performance of individual models vs the hybrid ensemble system.
    
    **Models compared:**
    - **TF-IDF Only**: Content-based filtering using text similarity
    - **RF Only**: Random Forest using metadata (7 features)
    - **Hybrid Ensemble**: Combination of all models with meta-learner
    """)
    
    # Try to load existing results first
    existing_results = load_comparison_results()
    
    if existing_results is not None:
        st.success("Loaded existing comparison results")
        st.session_state.comparison_results = existing_results
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Options")
        test_size = st.slider("Test Size", 100, 500, 200, 50,
                            help="Number of products to test")
        
        if st.button("🔄 Run Comparison", type="primary"):
            if not st.session_state.models_trained:
                st.error("⚠️ Train models first!")
            else:
                results = run_baseline_comparison(test_size)
                st.session_state.comparison_results = results
                results.to_csv('baseline_comparison_results.csv', index=False)
                st.success("✅ Results saved")
                st.rerun()
    
    with col2:
        if st.session_state.comparison_results is not None:
            results_df = st.session_state.comparison_results
            
            st.subheader("📋 Results Table")
            st.dataframe(results_df, use_container_width=True)
            
            # Create charts
            fig = create_comparison_charts(results_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Improvement chart
            imp_fig = create_improvement_chart(results_df)
            if imp_fig:
                st.plotly_chart(imp_fig, use_container_width=True)


def load_comparison_results():
    """Load baseline comparison results if available"""
    try:
        results = pd.read_csv('baseline_comparison_results.csv')
        return results
    except:
        return None


def run_baseline_comparison(test_size=200):
    """Run baseline comparison with smaller test set"""
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    
    from ensemble_system import EnsembleHybridSystem as HybridRecommenderSystem
    
    st.info("Running baseline comparison... This may take 1-2 minutes")
    
    df = st.session_state.df
    hybrid_system = HybridRecommenderSystem()
    
    test_indices = np.random.choice(len(df), min(test_size, len(df)), replace=False)
    y_test = df.iloc[test_indices]['average_rating'].values
    
    results = []
    progress_bar = st.progress(0)
    
    # TF-IDF Only
    progress_bar.progress(25)
    tfidf = TFIDFContentModel()
    if tfidf.load_model():
        predictions = []
        for idx in test_indices:
            score = tfidf.get_score(idx)
            predictions.append(score * 5.0)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({'model': 'TF-IDF Only', 'r2': r2, 'mae': mae, 'rmse': rmse})
    
    # RF Only
    progress_bar.progress(50)
    rf = RandomForestModel()
    if rf.load_model():
        predictions = []
        for idx in test_indices:
            score = rf.get_score(idx)
            predictions.append(score * 5.0)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({'model': 'RF Only', 'r2': r2, 'mae': mae, 'rmse': rmse})
    
    # Ensemble
    progress_bar.progress(75)
    if hybrid_system.load_trained_models():
        predictions = []
        for idx in test_indices:
            scores = hybrid_system.get_ensemble_score(idx)
            predictions.append(scores['ensemble_score'] * 5.0)
        
        r2 = r2_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        results.append({'model': 'Hybrid Ensemble', 'r2': r2, 'mae': mae, 'rmse': rmse})
    
    progress_bar.progress(100)
    
    return pd.DataFrame(results)


def create_comparison_charts(results_df):
    """Create interactive comparison charts"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('R² Score', 'MAE (Lower is Better)', 'RMSE (Lower is Better)'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
    )
    
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    fig.add_trace(
        go.Bar(x=results_df['model'], y=results_df['r2'], name='R²',
                marker_color=colors, text=results_df['r2'].round(3), textposition='outside', showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=results_df['model'], y=results_df['mae'], name='MAE',
                marker_color=colors, text=results_df['mae'].round(3), textposition='outside', showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=results_df['model'], y=results_df['rmse'], name='RMSE',
                marker_color=colors, text=results_df['rmse'].round(3), textposition='outside', showlegend=False),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False, title_text="Model Performance Comparison", title_font_size=20)
    fig.update_xaxes(tickangle=-45)
    
    return fig


def create_improvement_chart(results_df):
    """Create improvement percentage chart"""
    if 'Hybrid Ensemble' not in results_df['model'].values:
        return None
    
    ensemble_r2 = results_df[results_df['model'] == 'Hybrid Ensemble']['r2'].values[0]
    
    improvements = []
    for idx, row in results_df.iterrows():
        if row['model'] != 'Hybrid Ensemble' and row['r2'] > 0:
            improvement = ((ensemble_r2 - row['r2']) / row['r2']) * 100
            improvements.append({'model': f"vs {row['model']}", 'improvement': improvement})
    
    if not improvements:
        return None
    
    imp_df = pd.DataFrame(improvements)
    
    import plotly.express as px
    fig = px.bar(imp_df, x='improvement', y='model', orientation='h',
                title='Hybrid Ensemble Improvement (%)', color='improvement',
                color_continuous_scale='RdYlGn', text='improvement')
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(height=300, showlegend=False)
    
    return fig