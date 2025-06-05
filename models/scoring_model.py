import pandas as pd
import numpy as np
import joblib
import os

class StartupScoringModel:
    """
    Scoring model for startups based on team, product, market, and finance metrics.
    Uses a weighted approach to calculate an overall score from 0 to 100.
    """
    
    def __init__(self):
        """Initialize the scoring model with default weights"""
        # Define category weights
        self.weights = {
            'team': 0.25,
            'product': 0.30,
            'market': 0.20,
            'finance': 0.25
        }
        
        # Define feature weights within each category
        self.feature_weights = {
            'team': {
                'founder_count': 0.3,
                'founder_experience': 0.4,
                'previous_startups': 0.3
            },
            'product': {
                'product_stage': 0.3,
                'product_uniqueness': 0.4,
                'innovation_score': 0.3
            },
            'market': {
                'market_size': 0.3,
                'market_growth_rate': 0.3,
                'competitors_count': 0.2,
                'user_growth_rate': 0.2
            },
            'finance': {
                'revenue': 0.3,
                'burn_rate': 0.2,
                'total_investment': 0.2,
                'cash_reserves': 0.3
            }
        }
        
        # Define value mappings for categorical features
        self.value_mappings = {
            'product_stage': {
                'Idea': 0.2,
                'MVP': 0.4,
                'Growth': 0.6,
                'Scale': 0.8,
                'Maturity': 1.0
            }
        }
    
    def _normalize_value(self, value, feature_name, feature_data=None):
        """
        Normalize a feature value to a score between 0 and 1
        
        Parameters:
        -----------
        value : float or str
            The value to normalize
        feature_name : str
            The name of the feature
        feature_data : pandas.Series, optional
            Series containing all values for this feature (for min-max scaling)
            
        Returns:
        --------
        float
            Normalized value between 0 and 1
        """
        # Handle categorical features
        if feature_name in self.value_mappings:
            return self.value_mappings[feature_name].get(value, 0)
        
        # Handle numerical features
        if feature_data is not None:
            # Use min-max scaling for most features
            min_val = feature_data.min()
            max_val = feature_data.max()
            
            if max_val == min_val:
                return 0.5  # Default if all values are the same
            
            # Special case for features where lower is better
            if feature_name in ['burn_rate', 'competitors_count', 'risk_score']:
                return 1 - ((value - min_val) / (max_val - min_val))
            
            # For all other features, higher is better
            return (value - min_val) / (max_val - min_val)
        
        # Default normalization for binary features
        if feature_name == 'previous_startups':
            return 1.0 if value > 0 else 0.0
        
        return 0.0  # Default fallback
    
    def calculate_category_score(self, startup_data, category, all_data=None):
        """
        Calculate score for a specific category
        
        Parameters:
        -----------
        startup_data : pandas.Series
            Series containing startup data
        category : str
            Category to calculate score for ('team', 'product', 'market', or 'finance')
        all_data : pandas.DataFrame, optional
            DataFrame containing all startups data for normalization
            
        Returns:
        --------
        float
            Score for the category (0-100)
        """
        features = self.feature_weights[category]
        score = 0.0
        
        for feature, weight in features.items():
            if feature in startup_data:
                # Normalize the value
                if all_data is not None and feature in all_data:
                    normalized_value = self._normalize_value(startup_data[feature], feature, all_data[feature])
                else:
                    normalized_value = self._normalize_value(startup_data[feature], feature)
                
                # Add weighted score
                score += normalized_value * weight
        
        # Return score on a 0-100 scale
        return score * 100
    
    def calculate_overall_score(self, startup_data, all_data=None):
        """
        Calculate overall score for a startup
        
        Parameters:
        -----------
        startup_data : pandas.Series
            Series containing startup data
        all_data : pandas.DataFrame, optional
            DataFrame containing all startups data for normalization
            
        Returns:
        --------
        dict
            Dictionary containing overall score and category scores
        """
        category_scores = {}
        
        # Calculate score for each category
        for category in self.weights:
            category_scores[category] = self.calculate_category_score(startup_data, category, all_data)
        
        # Calculate overall score
        overall_score = 0.0
        for category, score in category_scores.items():
            overall_score += score * self.weights[category]
        
        # Determine risk category
        if overall_score >= 70:
            risk_category = "Strong Bet"
        elif overall_score >= 40:
            risk_category = "Medium"
        else:
            risk_category = "High Risk"
        
        return {
            'overall_score': overall_score,
            'category_scores': category_scores,
            'risk_category': risk_category
        }
    
    def score_startups(self, startups_df):
        """
        Score all startups in a DataFrame
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added score columns
        """
        # Create a copy of the input DataFrame
        result_df = startups_df.copy()
        
        # Calculate scores for each startup
        scores = []
        for _, startup in startups_df.iterrows():
            score_data = self.calculate_overall_score(startup, startups_df)
            scores.append(score_data)
        
        # Add score columns to the DataFrame
        result_df['overall_score'] = [s['overall_score'] for s in scores]
        result_df['risk_category'] = [s['risk_category'] for s in scores]
        
        for category in self.weights:
            result_df[f'{category}_score'] = [s['category_scores'][category] for s in scores]
        
        return result_df
    
    def save(self, filepath):
        """Save the model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        return joblib.load(filepath)
