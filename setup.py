import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data generator and model utils
from utils.data_generator import save_data
from utils.model_utils import initialize_models

def main():
    """
    Generate synthetic data and initialize models
    """
    print("Setting up StartIQ platform...")
    
    # Set paths relative to current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    models_dir = os.path.join(current_dir, 'models')
    
    # Generate data
    print("\nGenerating synthetic data...")
    startups_df, user_metrics_df = save_data(data_dir)
    
    # Initialize models
    print("\nInitializing and training models...")
    models, _, _ = initialize_models(data_dir, models_dir)
    
    print("\nSetup complete! You can now run the Streamlit app with:")
    print("streamlit run app.py")

if __name__ == "__main__":
    main()
