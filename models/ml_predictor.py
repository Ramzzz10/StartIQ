import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

class MLSuccessPredictionModel:
    """
    Machine Learning model for predicting startup success.
    Uses RandomForest or XGBoost to predict the 'success' target variable.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize the ML prediction model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('random_forest' or 'xgboost')
        """
        self.model_type = model_type
        self.model = None
        self.preprocessor = None
        self.feature_importances = None
        self.metrics = None
        self.classes_ = None
    
    def _preprocess_data(self, X):
        """
        Preprocess the data for training or prediction
        
        Parameters:
        -----------
        X : pandas.DataFrame
            DataFrame containing features
            
        Returns:
        --------
        numpy.ndarray
            Preprocessed features
        """
        if self.preprocessor is None:
            # Define categorical and numerical features
            categorical_features = ['country', 'industry', 'product_stage']
            numerical_features = [col for col in X.columns if col not in categorical_features + ['id', 'name', 'success']]
            
            # Create preprocessor
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            numerical_transformer = StandardScaler()
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            
            return self.preprocessor.fit_transform(X)
        else:
            return self.preprocessor.transform(X)
    
    def train(self, startups_df, test_size=0.2, random_state=42):
        """
        Train the ML model
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict
            Dictionary containing model metrics
        """
        # Filter out rows with 'Unclear' success label
        df = startups_df[startups_df['success'] != 'Unclear'].copy()
        
        # Split features and target
        X = df.drop(['success'], axis=1)
        y = df['success']
        
        # Store classes
        self.classes_ = np.unique(y)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Preprocess data
        X_train_processed = self._preprocess_data(X_train)
        X_test_processed = self._preprocess_data(X_test)
        
        # Create and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(random_state=random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics
        self.metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': report
        }
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = self.model.feature_importances_
        
        return self.metrics
    
    def predict(self, startup_data):
        """
        Predict success for a startup
        
        Parameters:
        -----------
        startup_data : pandas.DataFrame or pandas.Series
            Startup data to predict
            
        Returns:
        --------
        dict
            Dictionary containing prediction and probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to DataFrame if Series
        if isinstance(startup_data, pd.Series):
            startup_data = pd.DataFrame([startup_data])
        
        # Preprocess data
        X_processed = self._preprocess_data(startup_data)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Create result dictionary
        result = {
            'prediction': prediction,
            'probability': dict(zip(self.classes_, probabilities))
        }
        
        return result
    
    def predict_batch(self, startups_df):
        """
        Predict success for multiple startups
        
        Parameters:
        -----------
        startups_df : pandas.DataFrame
            DataFrame containing startup data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added prediction columns
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create a copy of the input DataFrame
        result_df = startups_df.copy()
        
        # Preprocess data
        X_processed = self._preprocess_data(startups_df)
        
        # Make predictions
        predictions = self.model.predict(X_processed)
        probabilities = self.model.predict_proba(X_processed)
        
        # Add prediction columns to the DataFrame
        result_df['predicted_success'] = predictions
        
        # Add probability columns
        for i, class_name in enumerate(self.classes_):
            result_df[f'prob_{class_name}'] = probabilities[:, i]
        
        return result_df
    
    def plot_confusion_matrix(self, figsize=(10, 8)):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing confusion matrix plot
        """
        if self.metrics is None:
            raise ValueError("Model not trained yet")
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes_, yticklabels=self.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names=None, top_n=10, figsize=(12, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : list, optional
            List of feature names
        top_n : int
            Number of top features to show
        figsize : tuple
            Figure size
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure containing feature importance plot
        """
        if self.feature_importances is None:
            raise ValueError("Model does not have feature importances")
        
        # Get feature importances
        importances = self.feature_importances
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importances))]
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1][:top_n]
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]
        
        # Plot
        plt.figure(figsize=figsize)
        plt.bar(range(len(top_importances)), top_importances, align='center')
        plt.xticks(range(len(top_importances)), top_names, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        
        return plt.gcf()
    
    def save(self, filepath):
        """Save the model to a file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load the model from a file"""
        return joblib.load(filepath)
