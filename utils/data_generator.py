import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import os

def generate_startup_data(num_samples=5000, seed=42):
    """
    Generate synthetic startup data
    
    Parameters:
    -----------
    num_samples : int
        Number of startup samples to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing synthetic startup data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    fake = Faker()
    Faker.seed(seed)
    
    # Define possible values for categorical features
    countries = ['USA', 'UK', 'Germany', 'France', 'India', 'China', 'Canada', 'Israel', 'Singapore', 'Australia']
    industries = ['AI', 'FinTech', 'HealthTech', 'EdTech', 'E-Commerce', 'SaaS', 'IoT', 'Blockchain', 'CleanTech', 'FoodTech']
    product_stages = ['Idea', 'MVP', 'Growth', 'Scale', 'Maturity']
    success_labels = ['Success', 'Fail', 'Unclear']
    
    # Generate data
    data = {
        'id': list(range(1, num_samples + 1)),
        'name': [fake.company() for _ in range(num_samples)],
        'year_founded': np.random.randint(2005, 2025, num_samples),
        'country': [random.choice(countries) for _ in range(num_samples)],
        'industry': [random.choice(industries) for _ in range(num_samples)],
        'founder_count': np.random.randint(1, 6, num_samples),
        'founder_experience': np.random.randint(0, 21, num_samples),
        'previous_startups': np.random.randint(0, 2, num_samples),
        'product_stage': [random.choice(product_stages) for _ in range(num_samples)],
        'product_uniqueness': np.random.randint(0, 11, num_samples),
        'competitors_count': np.random.randint(0, 11, num_samples),
        'revenue': np.random.uniform(0, 10000000, num_samples),
        'burn_rate': np.random.uniform(0, 2000000, num_samples),
        'total_investment': np.random.uniform(0, 20000000, num_samples),
        'active_users': np.random.randint(0, 1000000, num_samples),
        'user_growth_rate': np.random.uniform(0, 100, num_samples),
        'retention': np.random.uniform(0, 100, num_samples),
        'user_reviews': np.random.randint(0, 101, num_samples),
        'nps': np.random.randint(-100, 101, num_samples),
        'innovation_score': np.random.randint(0, 11, num_samples),
        'financial_stability': np.random.randint(0, 11, num_samples),
        'risk_score': np.random.randint(0, 11, num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate cash reserves based on burn rate and total investment
    df['cash_reserves'] = df['total_investment'] - (df['burn_rate'] * np.random.uniform(0, 3, num_samples))
    df['cash_reserves'] = df['cash_reserves'].clip(0)
    
    # Calculate market size and growth rate
    df['market_size'] = np.random.uniform(10000000, 1000000000, num_samples)
    df['market_growth_rate'] = np.random.uniform(0, 50, num_samples)
    
    # Add some correlation between features and success
    # Higher probability of success for startups with:
    # - More experienced founders
    # - Higher product uniqueness
    # - Better retention
    # - Higher NPS
    # - More financial stability
    
    success_score = (
        0.2 * df['founder_experience'] / 20 +
        0.2 * df['product_uniqueness'] / 10 +
        0.2 * df['retention'] / 100 +
        0.2 * (df['nps'] + 100) / 200 +
        0.2 * df['financial_stability'] / 10
    )
    
    # Convert score to probabilities
    probs = np.zeros((num_samples, 3))
    probs[:, 0] = success_score  # Success probability
    probs[:, 1] = 1 - success_score  # Fail probability
    probs[:, 2] = 0.2  # Unclear probability (constant)
    
    # Normalize probabilities
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    # Sample success labels based on probabilities
    df['success'] = [np.random.choice(success_labels, p=probs[i]) for i in range(num_samples)]
    
    # Add more correlations for realism
    # Startups in Idea stage have lower revenue and users
    idea_mask = df['product_stage'] == 'Idea'
    df.loc[idea_mask, 'revenue'] = df.loc[idea_mask, 'revenue'] * 0.1
    df.loc[idea_mask, 'active_users'] = df.loc[idea_mask, 'active_users'] * 0.1
    
    # Startups with high burn rate and low investment have lower financial stability
    high_burn_low_inv = (df['burn_rate'] > 1000000) & (df['total_investment'] < 5000000)
    df.loc[high_burn_low_inv, 'financial_stability'] = df.loc[high_burn_low_inv, 'financial_stability'] * 0.5
    
    # Round numerical values for better readability
    df['revenue'] = df['revenue'].round(2)
    df['burn_rate'] = df['burn_rate'].round(2)
    df['total_investment'] = df['total_investment'].round(2)
    df['cash_reserves'] = df['cash_reserves'].round(2)
    df['market_size'] = df['market_size'].round(2)
    df['user_growth_rate'] = df['user_growth_rate'].round(2)
    df['retention'] = df['retention'].round(2)
    df['market_growth_rate'] = df['market_growth_rate'].round(2)
    
    return df

def generate_user_metrics(startups_df, seed=42):
    """
    Generate user metrics data for startups
    
    Parameters:
    -----------
    startups_df : pandas.DataFrame
        DataFrame containing startup data
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing user metrics data
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    num_samples = len(startups_df)
    
    # Generate data
    data = {
        'startup_id': startups_df['id'].values,
        'retention_d1': np.random.uniform(30, 100, num_samples),
        'retention_d7': np.random.uniform(20, 90, num_samples),
        'retention_d30': np.random.uniform(10, 80, num_samples),
        'sessions_per_user': np.random.uniform(1, 20, num_samples),
        'time_in_app_minutes': np.random.uniform(1, 60, num_samples),
        'actions_per_session': np.random.uniform(1, 30, num_samples),
        'app_rating': np.random.uniform(1, 5, num_samples),
        'nps': startups_df['nps'].values,  # Copy from startups data
        'user_growth_rate': startups_df['user_growth_rate'].values,  # Copy from startups data
        'viral_coefficient': np.random.uniform(0, 2, num_samples),
        'organic_percentage': np.random.uniform(10, 100, num_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlations for realism
    # Better retention correlates with higher app rating
    df['app_rating'] = 0.7 * df['app_rating'] + 0.3 * (df['retention_d30'] / 20)
    df['app_rating'] = df['app_rating'].clip(1, 5)
    
    # Higher NPS correlates with better retention
    df['retention_d30'] = 0.8 * df['retention_d30'] + 0.2 * ((df['nps'] + 100) / 200 * 80)
    df['retention_d30'] = df['retention_d30'].clip(0, 100)
    
    # Higher viral coefficient correlates with higher organic percentage
    df['organic_percentage'] = 0.7 * df['organic_percentage'] + 0.3 * (df['viral_coefficient'] / 2 * 100)
    df['organic_percentage'] = df['organic_percentage'].clip(0, 100)
    
    # Round numerical values for better readability
    for col in df.columns:
        if col != 'startup_id':
            df[col] = df[col].round(2)
    
    return df

def save_data(output_dir='../data'):
    """
    Generate and save startup data and user metrics
    
    Parameters:
    -----------
    output_dir : str
        Directory to save data files
    
    Returns:
    --------
    tuple
        (startups_df, user_metrics_df)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate data
    startups_df = generate_startup_data()
    user_metrics_df = generate_user_metrics(startups_df)
    
    # Save data to CSV
    startups_df.to_csv(os.path.join(output_dir, 'startups_data.csv'), index=False)
    user_metrics_df.to_csv(os.path.join(output_dir, 'user_metrics.csv'), index=False)
    
    print(f"Generated {len(startups_df)} startup records and saved to {output_dir}")
    print(f"Generated {len(user_metrics_df)} user metrics records and saved to {output_dir}")
    
    return startups_df, user_metrics_df

if __name__ == "__main__":
    save_data()
