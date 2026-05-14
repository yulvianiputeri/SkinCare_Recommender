"""Tab 5: Analytics"""
import streamlit as st


def render_analytics(df):
    """Render analytics tab"""
    from analytics import SkincareAnalytics
    
    st.header("📈 Analytics Dashboard")
    
    analytics = SkincareAnalytics()
    analytics.df = df
    analytics.render_dashboard()