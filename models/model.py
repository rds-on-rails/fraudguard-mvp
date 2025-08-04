"""
Enhanced fraud detection model with improved data generation and training.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from faker import Faker
from datetime import datetime, timedelta
import uuid
import logging
from typing import Dict, List, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the trained model
_trained_model = None
_scaler = None
_label_encoders = {}
_feature_columns = None

def generate_dummy_data(n: int = 1000) -> pd.DataFrame:
    """
    Generate dummy transaction data for training and testing.
    
    Args:
        n: Number of transaction records to generate
        
    Returns:
        DataFrame with columns: transaction_id, amount, timestamp, location, device_type, merchant_id, user_id
    """
    fake = Faker()
    logger.info(f"Generating {n} dummy transaction records...")
    
    transactions = []
    
    # Generate some consistent patterns for realistic data
    users = [str(uuid.uuid4()) for _ in range(max(1, min(200, n // 5)))]
    locations = [fake.city() for _ in range(max(1, min(50, n // 20)))]
    device_types = ['mobile', 'desktop', 'tablet']
    merchants = [fake.company() for _ in range(max(1, min(100, n // 10)))]
    
    for _ in range(n):
        # Create mostly normal transactions with some outliers
        if np.random.random() < 0.05:  # 5% suspicious transactions
            amount = np.random.uniform(5000, 10000)  # High amounts
        else:
            amount = np.random.uniform(0.01, 1000)  # Normal amounts
        
        transaction = {
            'transaction_id': str(uuid.uuid4()),
            'amount': round(amount, 2),
            'timestamp': fake.date_time_between(
                start_date='-30d',
                end_date='now'
            ),
            'location': np.random.choice(locations),
            'device_type': np.random.choice(device_types),
            'merchant_id': np.random.choice(merchants),
            'user_id': np.random.choice(users)
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Generated {len(df)} transaction records successfully")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp to hour and encode categorical values (label encoding).
    Return only numerical features for training.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        DataFrame with only numerical features
    """
    return _extract_features(df)

def _extract_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer features from raw transaction data.
    
    Args:
        data: Raw transaction DataFrame
        
    Returns:
        DataFrame with engineered features ready for ML model
    """
    df = data.copy()
    
    # Convert timestamp to datetime if it's not already
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['timestamp_numeric'] = df['timestamp'].astype('int64') // 10**9  # Unix timestamp
    
    # Amount features
    df['amount_log'] = np.log(df['amount'] + 1)
    df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
    
    # User behavior features
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
    user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_transaction_count']
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Merchant behavior features
    merchant_stats = df.groupby('merchant_id')['amount'].agg(['count', 'mean']).reset_index()
    merchant_stats.columns = ['merchant_id', 'merchant_transaction_count', 'merchant_avg_amount']
    df = df.merge(merchant_stats, on='merchant_id', how='left')
    
    # Categorical encoding (using global label encoders for consistency)
    global _label_encoders
    
    categorical_cols = ['location', 'device_type', 'merchant_id', 'user_id']
    
    for col in categorical_cols:
        if col not in _label_encoders:
            _label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = _label_encoders[col].fit_transform(df[col].astype(str))
        else:
            # Handle new values not seen during training
            known_values = set(_label_encoders[col].classes_)
            df[f'{col}_mapped'] = df[col].astype(str).apply(
                lambda x: x if x in known_values else 'unknown'
            )
            
            # Add 'unknown' to encoder if not present
            if 'unknown' not in _label_encoders[col].classes_:
                _label_encoders[col].classes_ = np.append(_label_encoders[col].classes_, 'unknown')
            
            df[f'{col}_encoded'] = _label_encoders[col].transform(df[f'{col}_mapped'])
    
    # Select only numerical features for ML model
    feature_columns = [
        'amount', 'amount_log', 'amount_zscore', 'hour', 'day_of_week', 'is_weekend',
        'timestamp_numeric', 'user_avg_amount', 'user_std_amount', 'user_transaction_count',
        'merchant_transaction_count', 'merchant_avg_amount',
        'location_encoded', 'device_type_encoded', 'merchant_id_encoded', 'user_id_encoded'
    ]
    
    # Ensure all feature columns exist
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    features_df = df[feature_columns].fillna(0)
    
    return features_df

def train_model() -> Dict[str, Any]:
    """
    Train the Isolation Forest model using generated dummy data.
    
    Returns:
        Dictionary with training results and model info
    """
    global _trained_model, _scaler, _feature_columns
    
    logger.info("Training Isolation Forest model with generated dummy data...")
    
    try:
        # Generate training data
        data = generate_dummy_data(1000)
        logger.info(f"Generated {len(data)} training transactions")
        
        # Extract features
        features = _extract_features(data)
        _feature_columns = features.columns.tolist()
        
        # Scale features
        _scaler = StandardScaler()
        features_scaled = _scaler.fit_transform(features)
        
        # Train Isolation Forest
        _trained_model = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
            n_estimators=100
        )
        
        _trained_model.fit(features_scaled)
        
        # Calculate some training statistics
        predictions = _trained_model.predict(features_scaled)
        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions) * 100
        
        result = {
            "status": "success",
            "training_samples": len(data),
            "feature_count": len(_feature_columns),
            "anomaly_rate": round(anomaly_rate, 2),
            "contamination": 0.1,
            "model_type": "Isolation Forest"
        }
        
        logger.info(f"Model training completed successfully:")
        logger.info(f"  - Training samples: {result['training_samples']}")
        logger.info(f"  - Features: {result['feature_count']}")
        logger.info(f"  - Detected anomaly rate: {result['anomaly_rate']}%")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def predict_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict fraud for raw transaction data.
    Accepts raw transaction data (same schema as dummy) and returns fraud flags.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Original DataFrame with added fraud_flag column (1: fraud, 0: normal)
    """
    global _trained_model, _scaler, _feature_columns
    
    if _trained_model is None or _scaler is None:
        raise ValueError("Model not trained. Please train the model first.")
    
    logger.info(f"Predicting fraud for {len(df)} transactions...")
    
    try:
        
        # Extract features
        features = _extract_features(df)
        
        # Ensure we have the same features as training
        for col in _feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        features = features[_feature_columns]
        
        # Scale features
        features_scaled = _scaler.transform(features)
        
        # Make predictions
        predictions = _trained_model.predict(features_scaled)
        
        # Convert predictions to fraud flags (1: fraud, 0: normal)
        fraud_flags = [1 if pred == -1 else 0 for pred in predictions]
        
        # Add fraud_flag to original DataFrame
        result_df = df.copy()
        result_df['fraud_flag'] = fraud_flags
        
        fraud_count = sum(fraud_flags)
        logger.info(f"Fraud prediction completed: {fraud_count}/{len(result_df)} flagged as suspicious")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error during fraud prediction: {str(e)}")
        raise

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the current trained model.
    
    Returns:
        Dictionary with model information
    """
    global _trained_model, _feature_columns
    
    if _trained_model is None:
        return {
            "status": "not_trained",
            "model_type": None,
            "feature_count": 0,
            "features": []
        }
    
    return {
        "status": "trained",
        "model_type": "Isolation Forest",
        "feature_count": len(_feature_columns) if _feature_columns else 0,
        "features": _feature_columns or [],
        "contamination": _trained_model.contamination,
        "n_estimators": _trained_model.n_estimators
    }

def is_model_trained() -> bool:
    """Check if the model is trained and ready for predictions."""
    return _trained_model is not None and _scaler is not None
