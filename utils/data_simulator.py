"""
Data simulation utilities for generating synthetic transaction data.
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

fake = Faker()

def generate_transaction_data(num_transactions: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic transaction data for fraud detection.
    
    Args:
        num_transactions: Number of transactions to generate
        
    Returns:
        DataFrame with columns: user_id, amount, timestamp, location, device_id
    """
    transactions = []
    
    # Generate some consistent user_ids and device_ids for realism
    user_ids = [f"user_{i:06d}" for i in range(1, min(500, num_transactions // 2) + 1)]
    device_ids = [str(uuid.uuid4()) for _ in range(min(200, num_transactions // 5))]
    
    for _ in range(num_transactions):
        # Create mostly normal transactions with some outliers
        if random.random() < 0.05:  # 5% chance of suspicious transaction
            amount = random.uniform(5000, 50000)  # High amounts
        else:
            amount = random.uniform(1, 1000)  # Normal amounts
            
        transaction = {
            'user_id': random.choice(user_ids),
            'amount': round(amount, 2),
            'timestamp': fake.date_time_between(
                start_date='-30d', 
                end_date='now'
            ),
            'location': fake.city(),
            'device_id': random.choice(device_ids)
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def generate_normal_training_data(num_transactions: int = 2000) -> pd.DataFrame:
    """
    Generate normal transaction data for training the fraud detection model.
    This data should represent typical, non-fraudulent transactions.
    
    Args:
        num_transactions: Number of normal transactions to generate
        
    Returns:
        DataFrame with normal transaction patterns
    """
    transactions = []
    
    # Generate consistent users and devices for normal behavior
    user_ids = [f"user_{i:06d}" for i in range(1, min(800, num_transactions // 3) + 1)]
    device_ids = [str(uuid.uuid4()) for _ in range(min(300, num_transactions // 7))]
    
    for _ in range(num_transactions):
        # Generate only normal transaction patterns
        amount = random.uniform(1, 1000)  # Normal transaction amounts
        
        transaction = {
            'user_id': random.choice(user_ids),
            'amount': round(amount, 2),
            'timestamp': fake.date_time_between(
                start_date='-60d', 
                end_date='-30d'  # Historical data for training
            ),
            'location': fake.city(),
            'device_id': random.choice(device_ids)
        }
        transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for machine learning model.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Extract time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Amount features
    df['amount_log'] = df['amount'].apply(lambda x: np.log(x + 1))
    
    # User behavior features (simple aggregations)
    user_stats = df.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
    user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_transaction_count']
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)
    
    df = df.merge(user_stats, on='user_id', how='left')
    
    # Device features
    device_stats = df.groupby('device_id')['amount'].agg(['count']).reset_index()
    device_stats.columns = ['device_id', 'device_transaction_count']
    
    df = df.merge(device_stats, on='device_id', how='left')
    
    # Select numerical features for model
    feature_columns = [
        'amount', 'amount_log', 'hour', 'day_of_week', 'is_weekend',
        'user_avg_amount', 'user_std_amount', 'user_transaction_count',
        'device_transaction_count'
    ]
    
    return df[feature_columns].fillna(0)
