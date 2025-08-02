"""
Fraud detection model using Isolation Forest algorithm.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    """
    Fraud detection system using Isolation Forest for anomaly detection.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the fraud detector.
        
        Args:
            contamination: Expected proportion of outliers in the data
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def train_model(self, training_data: pd.DataFrame) -> None:
        """
        Train the fraud detection model on normal transaction data.
        
        Args:
            training_data: DataFrame with normal transaction features
        """
        logger.info(f"Training fraud detection model with {len(training_data)} transactions")
        
        # Store feature columns
        self.feature_columns = training_data.columns.tolist()
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(training_data)
        
        # Train the isolation forest
        self.model.fit(X_scaled)
        
        self.is_trained = True
        logger.info("Model training completed successfully")
        
    def predict(self, transaction_data: pd.DataFrame) -> List[int]:
        """
        Predict fraud flags for transaction data.
        
        Args:
            transaction_data: DataFrame with transaction features
            
        Returns:
            List of fraud flags (1 for fraud, 0 for normal)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if not all(col in transaction_data.columns for col in self.feature_columns):
            raise ValueError(f"Missing required features. Expected: {self.feature_columns}")
        
        # Select and order features correctly
        X = transaction_data[self.feature_columns]
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Predict anomalies (-1 for outlier, 1 for inlier)
        predictions = self.model.predict(X_scaled)
        
        # Convert to fraud flags (1 for fraud, 0 for normal)
        fraud_flags = [1 if pred == -1 else 0 for pred in predictions]
        
        return fraud_flags
    
    def predict_proba(self, transaction_data: pd.DataFrame) -> List[float]:
        """
        Get anomaly scores for transaction data.
        
        Args:
            transaction_data: DataFrame with transaction features
            
        Returns:
            List of anomaly scores (lower scores indicate more anomalous)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Select and order features correctly
        X = transaction_data[self.feature_columns]
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.decision_function(X_scaled)
        
        return scores.tolist()

# Global model instance
_fraud_model = None

def get_fraud_model() -> FraudDetector:
    """
    Get the global fraud detection model instance.
    
    Returns:
        FraudDetector instance
    """
    global _fraud_model
    if _fraud_model is None:
        _fraud_model = FraudDetector()
    return _fraud_model

def train_model() -> None:
    """
    Train the fraud detection model with synthetic normal transaction data.
    """
    from utils.data_simulator import generate_normal_training_data, prepare_features
    
    logger.info("Generating training data...")
    
    # Generate normal training data
    training_df = generate_normal_training_data(num_transactions=2000)
    
    # Prepare features
    training_features = prepare_features(training_df)
    
    # Get model instance and train
    model = get_fraud_model()
    model.train_model(training_features)
    
    logger.info("Model training completed and ready for predictions")

def predict_fraud(transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Predict fraud for a batch of transactions.
    
    Args:
        transactions: List of transaction dictionaries
        
    Returns:
        List of transaction dictionaries with fraud predictions
    """
    from utils.data_simulator import prepare_features
    
    # Convert to DataFrame
    df = pd.DataFrame(transactions)
    
    # Convert timestamp strings to datetime if needed
    if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Prepare features
    features = prepare_features(df)
    
    # Get model and predict
    model = get_fraud_model()
    if not model.is_trained:
        logger.warning("Model not trained. Training now with default data...")
        train_model()
    
    fraud_flags = model.predict(features)
    
    # Add fraud flags to original transactions
    result = []
    for i, transaction in enumerate(transactions):
        transaction_copy = transaction.copy()
        transaction_copy['fraud_flag'] = fraud_flags[i]
        result.append(transaction_copy)
    
    return result
