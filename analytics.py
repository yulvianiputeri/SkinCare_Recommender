import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

class SkincareAnalytics:
    """Simplified Analytics for Skincare Recommender"""
    
    def __init__(self):
        self.df = None
    
    def create_rating_chart(self):
        """Rating distribution chart"""
        fig = px.histogram(
            self.df, x='average_rating', nbins=20,
            title='📈 Rating Distribution',
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(showlegend=False, height=300)
        return fig
    
    def create_brand_chart(self):
        """Top brands chart"""
        top_brands = self.df['brand_name'].value_counts().head(10)
        
        fig = px.bar(
            x=top_brands.values, y=top_brands.index, orientation='h',
            title='🏆 Top 10 Brands',
            color=top_brands.values, color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False, height=350)
        return fig
    
    def create_category_chart(self):
        """Category distribution"""
        categories = self.df['default_category'].value_counts().head(8)
        
        fig = px.pie(
            values=categories.values, names=categories.index,
            title='🥧 Top Categories',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=350)
        return fig
    
    def create_engagement_chart(self):
        """Reviews vs Rating scatter"""
        # Sample for performance
        sample_df = self.df.sample(min(500, len(self.df)))
        
        fig = px.scatter(
            sample_df, x='total_reviews', y='average_rating',
            size='total_in_wishlist', color='average_rating',
            title='📈 Reviews vs Rating',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=350)
        return fig
    
    def get_insights(self):
        """Get key insights"""
        insights = {
            'total_products': len(self.df),
            'avg_rating': self.df['average_rating'].mean(),
            'total_brands': self.df['brand_name'].nunique(),
            'total_categories': self.df['default_category'].nunique(),
            'high_quality_pct': (self.df['average_rating'] >= 4.0).mean() * 100,
            'total_reviews': self.df['total_reviews'].sum(),
            'avg_reviews': self.df['total_reviews'].mean(),
            'top_brand': self.df['brand_name'].value_counts().index[0],
            'top_category': self.df['default_category'].value_counts().index[0]
        }
        return insights
    
    def render_dashboard(self):
        """Render complete dashboard"""
        if self.df is None:
            st.error("No data loaded")
            return
        
        # Key insights
        insights = self.get_insights()
        
        st.subheader("📊 Key Insights")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("High Quality Products", f"{insights['high_quality_pct']:.0f}%", 
                     help="Products with 4.0+ rating")
        with col2:
            st.metric("Avg Reviews/Product", f"{insights['avg_reviews']:.0f}")
        with col3:
            st.metric("Total Reviews", f"{insights['total_reviews']:,}")
        with col4:
            st.metric("Market Leaders", f"{insights['total_brands']} brands")
        
        # Charts
        st.subheader("📈 Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = self.create_rating_chart()
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            fig2 = self.create_brand_chart()
            st.plotly_chart(fig2, width='stretch')
        
        st.subheader("🎯 Market Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig3 = self.create_category_chart()
            st.plotly_chart(fig3, width='stretch')
        
        with col2:
            fig4 = self.create_engagement_chart()
            st.plotly_chart(fig4, width='stretch')
        
        # Top products
        st.subheader("🏆 Top Products")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🌟 Highest Rated**")
            top_rated = self.df.nlargest(5, 'average_rating')
            for i, (_, p) in enumerate(top_rated.iterrows(), 1):
                st.markdown(f"**{i}. {p['product_name'][:20]}...**")
                st.markdown(f"*{p['brand_name']}* | ⭐ {p['average_rating']:.2f}")
        
        with col2:
            st.markdown("**🔥 Most Reviewed**")
            most_reviewed = self.df.nlargest(5, 'total_reviews')
            for i, (_, p) in enumerate(most_reviewed.iterrows(), 1):
                st.markdown(f"**{i}. {p['product_name'][:20]}...**")
                st.markdown(f"*{p['brand_name']}* | 💬 {p['total_reviews']:,}")
        
        with col3:
            st.markdown("**❤️ Most Wishlisted**")
            most_wishlisted = self.df.nlargest(5, 'total_in_wishlist')
            for i, (_, p) in enumerate(most_wishlisted.iterrows(), 1):
                st.markdown(f"**{i}. {p['product_name'][:20]}...**")
                st.markdown(f"*{p['brand_name']}* | ❤️ {p['total_in_wishlist']:,}")
        
        # Summary stats
        st.subheader("📋 Dataset Summary")
        summary_data = {
            'Metric': ['Total Products', 'Unique Brands', 'Unique Categories', 
                        'Average Rating', 'Total Reviews', 'Top Brand', 'Top Category'],
            'Value': [f"{insights['total_products']:,}", 
                        f"{insights['total_brands']}", 
                        f"{insights['total_categories']}", 
                        f"{insights['avg_rating']:.2f}/5.0",
                        f"{insights['total_reviews']:,}",
                        insights['top_brand'],
                        insights['top_category']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True, width='stretch')


def main():
    """Test analytics"""
    analytics = SkincareAnalytics()
    
    try:
        analytics.df = pd.read_csv('dataset/processed/skincare_processed.csv')
        insights = analytics.get_insights()
        print("📊 Analytics Insights:")
        for key, value in insights.items():
            print(f"   {key}: {value}")
    except:
        print("❌ No data found for testing")

if __name__ == "__main__":
    main()