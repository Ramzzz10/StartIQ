import os
import joblib
import pandas as pd
from models.scoring_model import StartupScoringModel
from models.ml_predictor import MLSuccessPredictionModel
from models.pmf_analyzer import ProductMarketFitAnalyzer
from models.landscape_map import StartupLandscapeMap
from utils.data_generator import generate_startup_data, generate_user_metrics

def initialize_models(data_dir='../data', models_dir='../models'):
    """
    Initialize and train all models
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
    models_dir : str
        Directory to save trained models
        
    Returns:
    --------
    dict
        Dictionary containing trained models
    """
    # Create directories if they don't exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if data files exist, generate if not
    startups_path = os.path.join(data_dir, 'startups_data.csv')
    user_metrics_path = os.path.join(data_dir, 'user_metrics.csv')
    
    if not os.path.exists(startups_path) or not os.path.exists(user_metrics_path):
        print("Generating synthetic data...")
        startups_df = generate_startup_data()
        user_metrics_df = generate_user_metrics(startups_df)
        
        # Save data
        startups_df.to_csv(startups_path, index=False)
        user_metrics_df.to_csv(user_metrics_path, index=False)
        
        print(f"Generated {len(startups_df)} startup records and saved to {startups_path}")
        print(f"Generated {len(user_metrics_df)} user metrics records and saved to {user_metrics_path}")
    else:
        # Load existing data
        startups_df = pd.read_csv(startups_path)
        user_metrics_df = pd.read_csv(user_metrics_path)
        
        print(f"Loaded {len(startups_df)} startup records from {startups_path}")
        print(f"Loaded {len(user_metrics_df)} user metrics records from {user_metrics_path}")
    
    # Initialize models
    models = {}
    
    # Scoring Model
    scoring_model_path = os.path.join(models_dir, 'scoring_model.joblib')
    if os.path.exists(scoring_model_path):
        models['scoring_model'] = StartupScoringModel.load(scoring_model_path)
        print("Loaded existing Scoring Model")
    else:
        models['scoring_model'] = StartupScoringModel()
        models['scoring_model'].save(scoring_model_path)
        print("Initialized new Scoring Model")
    
    # ML Predictor
    ml_model_path = os.path.join(models_dir, 'ml_predictor.joblib')
    if os.path.exists(ml_model_path):
        models['ml_predictor'] = MLSuccessPredictionModel.load(ml_model_path)
        print("Loaded existing ML Predictor")
    else:
        models['ml_predictor'] = MLSuccessPredictionModel(model_type='random_forest')
        print("Training ML Predictor...")
        models['ml_predictor'].train(startups_df)
        models['ml_predictor'].save(ml_model_path)
        print("ML Predictor trained and saved")
    
    # PMF Analyzer
    pmf_analyzer_path = os.path.join(models_dir, 'pmf_analyzer.joblib')
    if os.path.exists(pmf_analyzer_path):
        models['pmf_analyzer'] = ProductMarketFitAnalyzer.load(pmf_analyzer_path)
        print("Loaded existing PMF Analyzer")
    else:
        models['pmf_analyzer'] = ProductMarketFitAnalyzer()
        models['pmf_analyzer'].save(pmf_analyzer_path)
        print("Initialized new PMF Analyzer")
    
    # Landscape Map
    landscape_map_path = os.path.join(models_dir, 'landscape_map.joblib')
    if os.path.exists(landscape_map_path):
        models['landscape_map'] = StartupLandscapeMap.load(landscape_map_path)
        print("Loaded existing Landscape Map")
    else:
        models['landscape_map'] = StartupLandscapeMap()
        models['landscape_map'].save(landscape_map_path)
        print("Initialized new Landscape Map")
    
    return models, startups_df, user_metrics_df

def load_data(data_dir='../data'):
    """
    Load data from CSV files
    
    Parameters:
    -----------
    data_dir : str
        Directory containing data files
        
    Returns:
    --------
    tuple
        (startups_df, user_metrics_df)
    """
    startups_path = os.path.join(data_dir, 'startups_data.csv')
    user_metrics_path = os.path.join(data_dir, 'user_metrics.csv')
    
    if not os.path.exists(startups_path) or not os.path.exists(user_metrics_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Run initialize_models() first.")
    
    startups_df = pd.read_csv(startups_path)
    user_metrics_df = pd.read_csv(user_metrics_path)
    
    return startups_df, user_metrics_df

def load_model(model_name, models_dir='../models'):
    """
    Load a specific model
    
    Parameters:
    -----------
    model_name : str
        Name of the model to load
    models_dir : str
        Directory containing model files
        
    Returns:
    --------
    object
        Loaded model
    """
    model_path = os.path.join(models_dir, f'{model_name}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Run initialize_models() first.")
    
    model = joblib.load(model_path)
    
    return model

def get_startup_by_id(startup_id, startups_df):
    """
    Get a startup by ID
    
    Parameters:
    -----------
    startup_id : int
        ID of the startup to get
    startups_df : pandas.DataFrame
        DataFrame containing startup data
        
    Returns:
    --------
    pandas.Series
        Startup data
    """
    startup = startups_df[startups_df['id'] == startup_id]
    
    if len(startup) == 0:
        raise ValueError(f"Startup with ID {startup_id} not found")
    
    return startup.iloc[0]

def get_user_metrics_by_startup_id(startup_id, user_metrics_df):
    """
    Get user metrics for a startup
    
    Parameters:
    -----------
    startup_id : int
        ID of the startup to get metrics for
    user_metrics_df : pandas.DataFrame
        DataFrame containing user metrics data
        
    Returns:
    --------
    pandas.Series
        User metrics data
    """
    metrics = user_metrics_df[user_metrics_df['startup_id'] == startup_id]
    
    if len(metrics) == 0:
        raise ValueError(f"User metrics for startup with ID {startup_id} not found")
    
    return metrics.iloc[0]
