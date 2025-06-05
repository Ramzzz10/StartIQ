import pandas as pd
import numpy as np
import joblib
import os

class ProductMarketFitAnalyzer:
    """
    Analyzer for Product-Market Fit (PMF) based on user metrics.
    Uses retention, NPS, user reviews, and growth metrics to determine PMF status.
    """
    
    def __init__(self):
        """Initialize the PMF analyzer with default thresholds"""
        # Define thresholds for PMF categories
        self.thresholds = {
            'high_pmf': {
                'retention': 60,  # Retention above 60%
                'nps': 40,        # NPS above 40
                'user_reviews': 80,  # User reviews above 80/100
                'user_growth_rate': 30  # Growth rate above 30%
            },
            'medium_pmf': {
                'retention': 30,  # Retention above 30%
                'nps': 0,         # NPS above 0
                'user_reviews': 60,  # User reviews above 60/100
                'user_growth_rate': 10  # Growth rate above 10%
            }
        }
        
        # Define weights for PMF score calculation
        self.weights = {
            'retention': 0.35,
            'nps': 0.25,
            'user_reviews': 0.20,
            'user_growth_rate': 0.20
        }
        
        # Define recommendation templates
        self.recommendations = {
            'low_retention': [
                "Improve user onboarding process to better explain product value",
                "Implement email re-engagement campaigns",
                "Add more engaging features to increase daily active usage",
                "Analyze drop-off points in the user journey and optimize them"
            ],
            'low_nps': [
                "Conduct user interviews to identify pain points",
                "Improve customer support response time",
                "Address common complaints in product roadmap",
                "Implement a feedback loop to show users their input matters"
            ],
            'low_reviews': [
                "Address critical bugs and usability issues",
                "Improve UI/UX design based on user feedback",
                "Implement gamification elements to improve user satisfaction",
                "Actively respond to negative reviews and show improvements"
            ],
            'low_growth': [
                "Optimize acquisition channels for better conversion",
                "Implement referral program to leverage existing users",
                "Explore new marketing channels",
                "Consider pricing model adjustments"
            ]
        }
    
    def calculate_pmf_score(self, startup_data, user_metrics=None):
        """
        Calculate PMF score for a startup
        
        Parameters:
        -----------
        startup_data : pandas.Series
            Series containing startup data
        user_metrics : pandas.Series, optional
            Series containing user metrics data
            
        Returns:
        --------
        dict
            Dictionary containing PMF score and category
        """
        # Combine startup data and user metrics if provided
        data = startup_data.copy()
        if user_metrics is not None:
            for col in user_metrics.index:
                if col != 'startup_id':
                    data[col] = user_metrics[col]
        
        # Calculate normalized scores for each metric
        scores = {}
        for metric, weight in self.weights.items():
            if metric in data:
                # Normalize the value based on the metric
                if metric == 'nps':
                    # NPS ranges from -100 to 100
                    normalized_value = (data[metric] + 100) / 200
                elif metric == 'retention':
                    # Retention is already a percentage (0-100)
                    normalized_value = data[metric] / 100
                elif metric == 'user_reviews':
                    # User reviews range from 0 to 100
                    normalized_value = data[metric] / 100
                elif metric == 'user_growth_rate':
                    # Cap growth rate at 100% for normalization
                    normalized_value = min(data[metric], 100) / 100
                else:
                    normalized_value = 0.5  # Default
                
                scores[metric] = normalized_value
        
        # Calculate weighted PMF score
        pmf_score = 0.0
        for metric, score in scores.items():
            pmf_score += score * self.weights.get(metric, 0)
        
        # Scale to 0-100
        pmf_score *= 100
        
        # Determine PMF category
        if pmf_score >= 70:
            pmf_category = "High PMF"
        elif pmf_score >= 40:
            pmf_category = "Medium PMF"
        else:
            pmf_category = "Low PMF"
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data)
        
        return {
            'pmf_score': pmf_score,
            'pmf_category': pmf_category,
            'metric_scores': scores,
            'recommendations': recommendations
        }
    
    def _generate_recommendations(self, data):
        """
        Generate recommendations based on metric values
        
        Parameters:
        -----------
        data : pandas.Series
            Series containing startup and user metrics data
            
        Returns:
        --------
        list
            List of recommendations
        """
        recommendations = []
        
        # Check each metric against thresholds
        if 'retention' in data and data['retention'] < self.thresholds['medium_pmf']['retention']:
            recommendations.extend(np.random.choice(self.recommendations['low_retention'], 2, replace=False))
        
        if 'nps' in data and data['nps'] < self.thresholds['medium_pmf']['nps']:
            recommendations.extend(np.random.choice(self.recommendations['low_nps'], 2, replace=False))
        
        if 'user_reviews' in data and data['user_reviews'] < self.thresholds['medium_pmf']['user_reviews']:
            recommendations.extend(np.random.choice(self.recommendations['low_reviews'], 2, replace=False))
        
        if 'user_growth_rate' in data and data['user_growth_rate'] < self.thresholds['medium_pmf']['user_growth_rate']:
            recommendations.extend(np.random.choice(self.recommendations['low_growth'], 2, replace=False))
        
        # If no specific recommendations, add general ones
        if not recommendations:
            recommendations = [
                "Continue monitoring key metrics to maintain PMF",
                "Consider expanding to adjacent markets",
                "Invest in scaling operations and infrastructure"
            ]
        
        return recommendations
    
    def analyze_startups(self, startups_df, user_metrics_df=None):
        """
        Analyze PMF for all startups in a DataFrame
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
        user_metrics_df : pandas.DataFrame, optional
            DataFrame containing user metrics data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added PMF columns
        """
        # Create a copy of the input DataFrame
        result_df = startups_df.copy()
        
        # Prepare user metrics dictionary for quick lookup
        user_metrics_dict = {}
        if user_metrics_df is not None:
            for _, row in user_metrics_df.iterrows():
                user_metrics_dict[row['startup_id']] = row
        
        # Calculate PMF for each startup
        pmf_data = []
        for _, startup in startups_df.iterrows():
            # Get user metrics if available
            user_metrics = user_metrics_dict.get(startup['id']) if user_metrics_dict else None
            
            # Calculate PMF
            pmf_result = self.calculate_pmf_score(startup, user_metrics)
            pmf_data.append(pmf_result)
        
        # Add PMF columns to the DataFrame
        result_df['pmf_score'] = [d['pmf_score'] for d in pmf_data]
        result_df['pmf_category'] = [d['pmf_category'] for d in pmf_data]
        
        # Add recommendations as a list column
        result_df['pmf_recommendations'] = [d['recommendations'] for d in pmf_data]
        
        return result_df
    
    def save(self, filepath):
        """Save the model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        return joblib.load(filepath)
