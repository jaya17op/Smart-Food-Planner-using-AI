"""
Smart Meal Planner - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import tempfile
import os
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

warnings.filterwarnings('ignore')
@dataclass
class MetricResults:
    """Data class to store recommendation performance metrics."""
    ctr: float
    precision: float
    recall: float
    f1_score: float
    engagement_rate: float
    avg_feedback_score: float
    total_recommendations: int
    total_clicks: int
    total_likes: int

class StreamlitRecommendationTracker:
    """Streamlit-optimized version of the recommendation tracker."""
    def __init__(self, df_users: pd.DataFrame, df_recipes: pd.DataFrame, df_recommendations: pd.DataFrame):
        self.df_users = df_users
        self.df_recipes = df_recipes
        self.df_recommendations = df_recommendations
        self.df_merged = None
        self.metrics = None
        # validation
        self._validate_data()
    
    def _validate_data(self) -> bool:
        """Validate uploaded data."""
        required_columns = {
            'users': ['user_id', 'feedback_score'],
            'recipes': ['recipe_id'],
            'recommendations': ['user_id', 'recipe_id', 'clicked', 'liked']
        }
        datasets = {
            'users': self.df_users,
            'recipes': self.df_recipes,
            'recommendations': self.df_recommendations
        }
        for dataset_name, df in datasets.items():
            missing_cols = set(required_columns[dataset_name]) - set(df.columns)
            if missing_cols:
                st.error(f"❌ Missing columns in {dataset_name}: {missing_cols}")
                return False
        
        return True
    def calculate_core_metrics(self) -> MetricResults:
        """Calculate core recommendation performance metrics."""
        df_recs = self.df_recommendations
        # Count
        total_recommendations = len(df_recs)
        total_clicks = int(df_recs['clicked'].sum())
        total_likes = int(df_recs['liked'].sum())
        # ctr
        ctr = total_clicks / total_recommendations if total_recommendations > 0 else 0
        # precision
        clicked_recs = df_recs[df_recs['clicked'] == 1]
        precision = (clicked_recs['liked'].sum() / len(clicked_recs) 
                    if len(clicked_recs) > 0 else 0)
        
        recall = total_likes / total_recommendations if total_recommendations > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall) 
                   if (precision + recall) > 0 else 0)
        engagement_rate = (total_clicks + total_likes) / (2 * total_recommendations)
        avg_feedback_score = self.df_users['feedback_score'].mean()
        
        self.metrics = MetricResults(
            ctr=ctr,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            engagement_rate=engagement_rate,
            avg_feedback_score=avg_feedback_score,
            total_recommendations=total_recommendations,
            total_clicks=total_clicks,
            total_likes=total_likes
        )
        return self.metrics
    def analyze_user_segments(self) -> pd.DataFrame:
        """Analyze performance across different user feedback segments."""
        self.df_merged = self.df_recommendations.merge(self.df_users, on='user_id')
        
        segment_analysis = self.df_merged.groupby('feedback_score').agg({
            'clicked': ['count', 'sum', 'mean'],
            'liked': ['sum', 'mean'],
            'user_id': 'nunique'
        }).round(4)
        
        segment_analysis.columns = [
            'total_recommendations', 'total_clicks', 'ctr',
            'total_likes', 'like_rate', 'unique_users'
        ]
        return segment_analysis
    
    def get_top_performers(self, top_n: int = 5) -> Dict[str, pd.Series]:
        """Get top performing recipes and users."""
        top_recipes = (self.df_recommendations[self.df_recommendations['liked'] == 1]
                      ['recipe_id'].value_counts().head(top_n))
        top_users = (self.df_recommendations[self.df_recommendations['clicked'] == 1]
                    ['user_id'].value_counts().head(top_n))
        return {'top_recipes': top_recipes, 'top_users': top_users}

st.set_page_config(
    page_title="AI Recommendation Performance Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .performance-excellent { border-left-color: #28a745; }
    .performance-good { border-left-color: #ffc107; }
    .performance-poor { border-left-color: #dc3545; }
    .sidebar-info {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🍽️ Smart Meal Planner</h1>
    <h2>AI Recommendation Performance Tracker</h2>
    <p>Professional analytics dashboard for optimizing meal recommendation algorithms</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 📁 Data Upload")
st.sidebar.markdown("Upload your CSV files to begin analysis")

# File uploaders
users_file = st.sidebar.file_uploader("👥 Users Data", type="csv", help="Upload users.csv with user_id and feedback_score columns")
recipes_file = st.sidebar.file_uploader("🍳 Recipes Data", type="csv", help="Upload recipes.csv with recipe_id column")
recs_file = st.sidebar.file_uploader("💡 Recommendations Data", type="csv", help="Upload recommendations.csv with user_id, recipe_id, clicked, and liked columns")

#application logic
if users_file and recipes_file and recs_file:
    try:
        with st.spinner("Loading data..."):
            users = pd.read_csv(users_file)
            recipes = pd.read_csv(recipes_file)
            recommendations = pd.read_csv(recs_file)
        tracker = StreamlitRecommendationTracker(users, recipes, recommendations)
        metrics = tracker.calculate_core_metrics()
        st.success("✅ Data loaded and analyzed successfully!")
       
        # data overview
        with st.expander("📊 Data Overview", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("👥 Users", f"{len(users):,}")
            with col2:
                st.metric("🍳 Recipes", f"{len(recipes):,}")
            with col3:
                st.metric("💡 Recommendations", f"{len(recommendations):,}")
        
        st.markdown("## 📈 Performance Metrics Dashboard")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_ctr = "🔥 Excellent" if metrics.ctr >= 0.10 else "⚠️ Needs work" if metrics.ctr < 0.02 else "✅ Good"
            st.metric("Click-Through Rate", f"{metrics.ctr:.2%}", delta=delta_ctr)
        with col2:
            delta_precision = "🎯 High Quality" if metrics.precision >= 0.70 else "📊 Moderate" if metrics.precision >= 0.50 else "🔧 Optimize"
            st.metric("Precision", f"{metrics.precision:.2%}", delta=delta_precision)
        with col3:
            st.metric("Recall", f"{metrics.recall:.2%}")
        with col4:
            delta_f1 = "⚖️ Balanced" if metrics.f1_score >= 0.60 else "🔄 Tune Algorithm" if metrics.f1_score < 0.40 else "📈 Improving"
            st.metric("F1 Score", f"{metrics.f1_score:.2%}", delta=delta_f1)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Engagement Rate", f"{metrics.engagement_rate:.2%}")
        with col2:
            st.metric("Avg Feedback Score", f"{metrics.avg_feedback_score:.2f}")
        with col3:
            st.metric("Total Interactions", f"{metrics.total_clicks + metrics.total_likes:,}")
        
        st.markdown("## 🎯 Performance Assessment")
        
        assessment_col1, assessment_col2 = st.columns(2)
        
        with assessment_col1:
            if metrics.ctr >= 0.10:
                st.success("🔥 **CTR**: Excellent - Users are highly engaged!")
            elif metrics.ctr >= 0.05:
                st.info("✅ **CTR**: Good - Solid user engagement")
            elif metrics.ctr >= 0.02:
                st.warning("⚠️ **CTR**: Fair - Room for improvement")
            else:
                st.error("❌ **CTR**: Poor - Needs optimization")
        with assessment_col2:
            if metrics.precision >= 0.70:
                st.success("🎯 **Precision**: Excellent - High quality recommendations")
            elif metrics.precision >= 0.50:
                st.info("📊 **Precision**: Good - Mostly relevant recommendations")
            else:
                st.warning("🔧 **Precision**: Needs improvement - Refine algorithm")
        #visualizations
        st.markdown("## 📊 Analytics Visualizations")
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "👥 User Segments", "🏆 Top Performers", "📋 Detailed Analysis"])
        
        with tab1:
            # performance
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Core Metrics', 'Engagement Distribution', 'CTR vs Feedback Score', 'Volume Metrics'),
                specs=[[{"type": "bar"}, {"type": "histogram"}],
                       [{"type": "scatter"}, {"type": "bar"}]]
            )
            metrics_data = ['CTR', 'Precision', 'Recall', 'F1 Score']
            metrics_values = [metrics.ctr, metrics.precision, metrics.recall, metrics.f1_score]
            
            fig.add_trace(
                go.Bar(x=metrics_data, y=metrics_values, name="Core Metrics", 
                       marker_color=['#667eea', '#764ba2', '#667eea', '#764ba2']),
                row=1, col=1
            )
            merged = recommendations.merge(users, on='user_id')
            engagement_data = merged.groupby('user_id').agg({'clicked': 'sum', 'liked': 'sum'}).reset_index()
            
            fig.add_trace(
                go.Histogram(x=engagement_data['clicked'], name="Clicks", opacity=0.7),
                row=1, col=2
            )
            # ctr vs feedback
            segment_data = tracker.analyze_user_segments().reset_index()
            fig.add_trace(
                go.Scatter(x=segment_data['feedback_score'], y=segment_data['ctr'], 
                          mode='lines+markers', name="CTR by Feedback"),
                row=2, col=1
            )
            
            #volume metrics
            volume_data = ['Recommendations', 'Clicks', 'Likes']
            volume_values = [metrics.total_recommendations, metrics.total_clicks, metrics.total_likes]
            
            fig.add_trace(
                go.Bar(x=volume_data, y=volume_values, name="Volume Metrics",
                       marker_color=['#28a745', '#ffc107', '#dc3545']),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=True, title_text="Performance Dashboard")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 👥 User Segment Analysis")
            segment_analysis = tracker.analyze_user_segments()
            
            #table
            st.dataframe(segment_analysis.style.format({
                'ctr': '{:.2%}',
                'like_rate': '{:.2%}',
                'total_recommendations': '{:,}',
                'total_clicks': '{:,}',
                'total_likes': '{:,}',
                'unique_users': '{:,}'
            }))
            
            # visualization
            fig_segments = px.bar(
                segment_analysis.reset_index(),
                x='feedback_score',
                y='ctr',
                title='Click-Through Rate by User Feedback Score',
                labels={'feedback_score': 'Feedback Score', 'ctr': 'Click-Through Rate'},
                color='ctr',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_segments, use_container_width=True)
        
        with tab3:
            st.markdown("### 🏆 Top Performers")
            top_performers = tracker.get_top_performers()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🍽️ Top Liked Recipes")
                top_recipes_df = pd.DataFrame({
                    'Recipe ID': top_performers['top_recipes'].index,
                    'Likes': top_performers['top_recipes'].values
                })
                st.dataframe(top_recipes_df, use_container_width=True)
                fig_recipes = px.bar(
                    top_recipes_df,
                    x='Recipe ID',
                    y='Likes',
                    title='Top 5 Most Liked Recipes',
                    color='Likes',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_recipes, use_container_width=True)
            
            with col2:
                st.markdown("#### 👤 Most Engaged Users")
                top_users_df = pd.DataFrame({
                    'User ID': top_performers['top_users'].index,
                    'Clicks': top_performers['top_users'].values
                })
                st.dataframe(top_users_df, use_container_width=True)
                fig_users = px.bar(
                    top_users_df,
                    x='User ID',
                    y='Clicks',
                    title='Top 5 Most Engaged Users',
                    color='Clicks',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_users, use_container_width=True)
        
        with tab4:
            st.markdown("### 📋 Detailed Analysis")
            #summary
            st.markdown("#### 😊 User Feedback Distribution")
            feedback_stats = users['feedback_score'].describe()
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(feedback_stats.to_frame('Statistics'))
            
            with col2:
                fig_feedback = px.histogram(
                    users,
                    x='feedback_score',
                    nbins=20,
                    title='User Feedback Score Distribution',
                    labels={'feedback_score': 'Feedback Score', 'count': 'Number of Users'}
                )
                st.plotly_chart(fig_feedback, use_container_width=True)
            
            #analysis
            st.markdown("#### 🔗 Correlation Analysis")
            merged_full = recommendations.merge(users, on='user_id')
            correlation_data = merged_full[['clicked', 'liked', 'feedback_score']].corr()
            
            fig_corr = px.imshow(
                correlation_data,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix: Clicks, Likes, and Feedback",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        #exports
        st.markdown("## 💾 Export Results")
        col1,col2 = st.columns(2)
        with col1:
            if st.button("📊 Generate Report"):
                report_data = {
                    'Metric': ['CTR', 'Precision', 'Recall', 'F1_Score', 'Engagement_Rate', 
                              'Avg_Feedback_Score', 'Total_Recommendations', 'Total_Clicks', 'Total_Likes'],
                    'Value': [
                        metrics.ctr, metrics.precision, metrics.recall, metrics.f1_score,
                        metrics.engagement_rate, metrics.avg_feedback_score,
                        metrics.total_recommendations, metrics.total_clicks, metrics.total_likes
                    ]
                }
                
                report_df = pd.DataFrame(report_data)
                
                #conversion of reports
                csv = report_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Performance Report",
                    data=csv,
                    file_name=f"recommendation_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export Segment Analysis"):
                segment_csv = tracker.analyze_user_segments().to_csv()
                st.download_button(
                    label="📥 Download Segment Analysis",
                    data=segment_csv,
                    file_name=f"user_segment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        #footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
            <p>🍽️ Smart Meal Planner - AI Recommendation Performance Tracker</p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.info("Please check that your CSV files have the correct format and required columns.")

else:
    st.markdown("## 🚀 Welcome to the AI Recommendation Performance Tracker!")
    
    st.markdown("""
    This professional dashboard helps you analyze and optimize your meal recommendation algorithms.
    
    ### 📋 Getting Started:
    1. **Upload your data files** using the sidebar
    2. **View comprehensive metrics** and performance indicators
    3. **Analyze user segments** and top performers
    4. **Export detailed reports** for stakeholder review
    
    ### 🎯 Key Features:
    - **Advanced Metrics**: CTR, Precision, Recall, F1 Score, Engagement Rate
    - **User Segmentation**: Performance analysis by feedback scores
    - **Interactive Visualizations**: Professional charts and dashboards
    - **Export Capabilities**: Download reports and analysis results
    - **Real-time Analysis**: Instant insights as you upload data
    """)
    
    # Sample data structure
    with st.expander("📊 Sample Data Structure", expanded=False):
        st.markdown("### Required CSV File Formats:")
        
        st.markdown("**users.csv:**")
        st.code("""
user_id,feedback_score
1,4.5
2,3.8
3,4.2
        """)
        
        st.markdown("**recipes.csv:**")
        st.code("""
recipe_id,recipe_name,category
R001,Pasta Carbonara,Italian
R002,Chicken Curry,Indian
R003,Caesar Salad,Healthy
        """)
        
        st.markdown("**recommendations.csv:**")
        st.code("""
user_id,recipe_id,clicked,liked
1,R001,1,1
1,R002,0,0
2,R001,1,0
        """)
    
    st.info("👈 Upload your CSV files in the sidebar to begin analysis!")