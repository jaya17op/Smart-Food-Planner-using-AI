import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
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


class RecommendationPerformanceTracker:
    """
    Smart AI Recommendation Performance Tracker for Smart Meal Planner
    """
    
    def __init__(self, users_path: str = 'users.csv', 
                 recipes_path: str = 'recipes.csv', 
                 recommendations_path: str = 'recommendations.csv'):
        """
        Initialize the performance tracker with data file paths.
        Args:
            users_path (str): Path to users CSV file
            recipes_path (str): Path to recipes CSV file  
            recommendations_path (str): Path to recommendations CSV file
        """
        self.users_path = users_path
        self.recipes_path = recipes_path
        self.recommendations_path = recommendations_path
        self.df_users: Optional[pd.DataFrame] = None
        self.df_recipes: Optional[pd.DataFrame] = None
        self.df_recommendations: Optional[pd.DataFrame] = None
        self.df_merged: Optional[pd.DataFrame] = None
        
        # Performance metrics
        self.metrics: Optional[MetricResults] = None
        
        # Load and validate data
        self._load_data()
        self._validate_data()
        
        logger.info("Recommendation PerformanceTracker initialized successfully")
    
    def _load_data(self) -> None:
        """Load CSV files into pandas DataFrames with error handling."""
        try:
            self.df_users = pd.read_csv(self.users_path)
            self.df_recipes = pd.read_csv(self.recipes_path)
            self.df_recommendations = pd.read_csv(self.recommendations_path)
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  - Users: {len(self.df_users)} records")
            logger.info(f"  - Recipes: {len(self.df_recipes)} records")
            logger.info(f"  - Recommendations: {len(self.df_recommendations)} records")
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _validate_data(self) -> None:
        """Validate data integrity and required columns."""
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
                raise ValueError(f"Missing columns in {dataset_name}: {missing_cols}")
        if self.df_recommendations['clicked'].isnull().any():
            logger.warning("Found null values in 'clicked' column")
        
        if self.df_recommendations['liked'].isnull().any():
            logger.warning("Found null values in 'liked' column")
        
        logger.info("Data validation completed successfully")
    
    def calculate_core_metrics(self) -> MetricResults:
        """
        Calculate core recommendation performance metrics.
        Returns:
            MetricResults: Comprehensive metrics including CTR, precision, recall, F1
        """
        df_recs = self.df_recommendations
        total_recommendations = len(df_recs)
        total_clicks = df_recs['clicked'].sum()
        total_likes = df_recs['liked'].sum()
        
        # Click through rate
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
    def generate_performance_report(self) -> None:
        """Generate and display comprehensive performance report."""
        if self.metrics is None:
            self.calculate_core_metrics()
        
        print("="*60)
        print("🍽️  SMART MEAL PLANNER - AI RECOMMENDATION PERFORMANCE REPORT")
        print("="*60)
        print(f"📊 Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("📈 CORE METRICS:")
        print(f"  • Click-Through Rate (CTR):     {self.metrics.ctr:.2%}")
        print(f"  • Precision:                   {self.metrics.precision:.2%}")
        print(f"  • Recall:                      {self.metrics.recall:.2%}")
        print(f"  • F1 Score:                    {self.metrics.f1_score:.2%}")
        print(f"  • Engagement Rate:             {self.metrics.engagement_rate:.2%}")
        print(f"  • Average Feedback Score:      {self.metrics.avg_feedback_score:.2f}")
        print()
        
        print("📊 VOLUME METRICS:")
        print(f"  • Total Recommendations:       {self.metrics.total_recommendations:,}")
        print(f"  • Total Clicks:                {self.metrics.total_clicks:,}")
        print(f"  • Total Likes:                 {self.metrics.total_likes:,}")
        print()
        
        # Performance assessment
        self._assess_performance()
        print("="*60)
    def _assess_performance(self) -> None:
        """Provide intelligent performance assessment and recommendations."""
        print("🎯 PERFORMANCE ASSESSMENT:")
        
        # CTR Assessment
        if self.metrics.ctr >= 0.10:
            print("  ✅ CTR: Excellent - Users are highly engaged with recommendations")
        elif self.metrics.ctr >= 0.05:
            print("  ✅ CTR: Good - Solid user engagement")
        elif self.metrics.ctr >= 0.02:
            print("  ⚠️  CTR: Fair - Room for improvement in recommendation relevance")
        else:
            print("  ❌ CTR: Poor - Recommendations need significant optimization")

            #precision
        if self.metrics.precision >= 0.70:
            print("  ✅ Precision: Excellent - High quality recommendations")
        elif self.metrics.precision >= 0.50:
            print("  ✅ Precision: Good - Recommendations are mostly relevant")
        else:
            print("  ⚠️ Precision: Needs improvement - Consider refining recommendation algorithm")
        
        # F1 Score Assessment
        if self.metrics.f1_score >= 0.60:
            print("  ✅ F1 Score: Well-balanced precision and recall")
        elif self.metrics.f1_score >= 0.40:
            print("  ⚠️  F1 Score: Moderate balance - Consider optimizing both precision and recall")
        else:
            print("  ❌ F1 Score: Poor balance - Algorithm needs significant tuning")
        
        print()
    
    def analyze_user_segments(self) -> pd.DataFrame:
        """Analyze performance across different user feedback segments."""
        # Merge recommendations with user data
        self.df_merged = self.df_recommendations.merge(self.df_users, on='user_id')
        
        # Group by feedback score and calculate metrics
        segment_analysis = self.df_merged.groupby('feedback_score').agg({
            'clicked': ['count', 'sum', 'mean'],
            'liked': ['sum', 'mean'],
            'user_id': 'nunique'
        }).round(4)
        # Flatten column names
        segment_analysis.columns = [
            'total_recommendations', 'total_clicks', 'ctr',
            'total_likes', 'like_rate', 'unique_users'
        ]
        return segment_analysis
    
    def get_top_performers(self, top_n: int = 5) -> Dict[str, pd.Series]:
        """
        Identify top performing recipes and most engaged users.
        
        Args:
            top_n (int): Number of top performers to return
            
        Returns:
            Dict containing top recipes and users
        """
        top_recipes = (self.df_recommendations[self.df_recommendations['liked'] == 1]
                      ['recipe_id'].value_counts().head(top_n))
        
        top_users = (self.df_recommendations[self.df_recommendations['clicked'] == 1]
                    ['user_id'].value_counts().head(top_n))
        
        return {
            'top_recipes': top_recipes,
            'top_users': top_users
        }
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations for performance analysis."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🍽️ Smart Meal Planner - AI Recommendation Performance Dashboard', 
                    fontsize=16, fontweight='bold')
        
        # 1. CTR by Feedback Score
        segment_data = self.analyze_user_segments()
        sns.barplot(x=segment_data.index, y=segment_data['ctr'], ax=axes[0, 0])
        axes[0, 0].set_title('Click-Through Rate by User Feedback Score')
        axes[0, 0].set_xlabel('Feedback Score')
        axes[0, 0].set_ylabel('CTR')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Metrics Overview
        metrics_data = {
            'CTR': self.metrics.ctr,
            'Precision': self.metrics.precision,
            'Recall': self.metrics.recall,
            'F1 Score': self.metrics.f1_score,
            'Engagement': self.metrics.engagement_rate
        }
        
        bars = axes[0, 1].bar(metrics_data.keys(), metrics_data.values())
        axes[0, 1].set_title('Core Performance Metrics')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2%}', ha='center', va='bottom')
        
        # 3. User Engagement Distribution
        engagement_data = self.df_merged.groupby('user_id').agg({
            'clicked': 'sum',
            'liked': 'sum'
        })
        
        axes[1, 0].hist(engagement_data['clicked'], bins=20, alpha=0.7, label='Clicks')
        axes[1, 0].hist(engagement_data['liked'], bins=20, alpha=0.7, label='Likes')
        axes[1, 0].set_title('User Engagement Distribution')
        axes[1, 0].set_xlabel('Number of Interactions')
        axes[1, 0].set_ylabel('Number of Users')
        axes[1, 0].legend()
        
        #Feedback Score Distribution
        axes[1, 1].hist(self.df_users['feedback_score'], bins=15, alpha=0.7, color='skyblue')
        axes[1, 1].set_title('User Feedback Score Distribution')
        axes[1, 1].set_xlabel('Feedback Score')
        axes[1, 1].set_ylabel('Number of Users')
        
        plt.tight_layout()
        plt.show()
    
    def export_results(self, filename: str = 'recommendation_performance_report.csv') -> None:
        """Export analysis results to CSV file."""
        if self.metrics is None:
            self.calculate_core_metrics()
        
        # Create summary dataframe
        summary_data = {
            'Metric': ['CTR', 'Precision', 'Recall', 'F1_Score', 'Engagement_Rate', 
                      'Avg_Feedback_Score', 'Total_Recommendations', 'Total_Clicks', 'Total_Likes'],
            'Value': [
                self.metrics.ctr, self.metrics.precision, self.metrics.recall,
                self.metrics.f1_score, self.metrics.engagement_rate,
                self.metrics.avg_feedback_score, self.metrics.total_recommendations,
                self.metrics.total_clicks, self.metrics.total_likes
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filename, index=False)
        logger.info(f"Results exported to {filename}")

    def run_complete_analysis(self) -> None:
        """Run complete analysis pipeline with all features."""
        logger.info("Starting complete recommendation performance analysis...")
        # Calculate metrics
        self.calculate_core_metrics()
        # Generate report
        self.generate_performance_report()
        # Analyze user segments
        print("🔍 USER SEGMENT ANALYSIS:")
        segment_analysis = self.analyze_user_segments()
        print(segment_analysis)
        print()
        
        #performance
        top_performers = self.get_top_performers()
        print("🏆 TOP PERFORMERS:")
        print("Top Liked Recipes:")
        print(top_performers['top_recipes'])
        print("\nMost Engaged Users:")
        print(top_performers['top_users'])
        print()
        
        #visualization
        self.create_visualizations()
        self.export_results()
        
        logger.info("Complete analysis finished successfully!")

if __name__ == "__main__":
    tracker = RecommendationPerformanceTracker()
    tracker.run_complete_analysis()
    
    print("\n" + "="*60)
    print("🔬 ADVANCED ANALYSIS EXAMPLES")
    print("="*60)
    
    #metrics
    custom_metrics = tracker.calculate_core_metrics()
    print(f"Custom F1 Score: {custom_metrics.f1_score:.4f}")
    
    #segment analysis
    segment_data = tracker.analyze_user_segments()
    high_feedback_users = segment_data.loc[segment_data.index >= 4.0]
    print(f"\nHigh Feedback Users (≥4.0) CTR: {high_feedback_users['ctr'].mean():.2%}")