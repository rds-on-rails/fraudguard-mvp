"""
Helper utilities for the Streamlit frontend.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """
    Check if the backend API is available and healthy.
    
    Returns:
        bool: True if API is healthy, False otherwise
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False

def get_api_info() -> Optional[Dict[str, Any]]:
    """
    Get API information and status.
    
    Returns:
        Dictionary with API info or None if failed
    """
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Failed to get API info: {e}")
        return None

def generate_sample_data(n: int = 100) -> Optional[pd.DataFrame]:
    """
    Generate sample transaction data using the backend API.
    
    Args:
        n: Number of records to generate
        
    Returns:
        DataFrame with generated transactions or None if failed
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-data",
            params={"n": n},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                df = pd.DataFrame(data["transactions"])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
        
        logger.error(f"Failed to generate sample data: {response.text}")
        return None
        
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        return None

def predict_fraud(transactions_df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Send transactions to the backend for fraud prediction.
    
    Args:
        transactions_df: DataFrame with transaction data
        
    Returns:
        Prediction results dictionary or None if failed
    """
    try:
        # Convert DataFrame to the required format
        transactions_list = []
        for _, row in transactions_df.iterrows():
            transaction = {
                "user_id": str(row['user_id']),
                "amount": float(row['amount']),
                "timestamp": row['timestamp'].isoformat(),
                "location": str(row['location']),
                "device_id": str(row['device_id'])
            }
            transactions_list.append(transaction)
        
        payload = {"transactions": transactions_list}
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Prediction API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error during fraud prediction: {e}")
        return None

def validate_transaction_data(df: pd.DataFrame) -> tuple[bool, List[str]]:
    """
    Validate that the uploaded data has the required columns and format.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    required_columns = ['user_id', 'amount', 'timestamp', 'location', 'device_id']
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    # Check data types and values
    if 'amount' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['amount']):
            errors.append("'amount' column must contain numeric values")
        elif (df['amount'] <= 0).any():
            errors.append("'amount' column must contain positive values")
    
    if 'timestamp' in df.columns:
        try:
            pd.to_datetime(df['timestamp'])
        except:
            errors.append("'timestamp' column must contain valid datetime values")
    
    # Check for empty values
    for col in required_columns:
        if col in df.columns and df[col].isna().any():
            errors.append(f"'{col}' column contains empty values")
    
    return len(errors) == 0, errors

def format_prediction_results(prediction_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Format prediction results for display in Streamlit.
    
    Args:
        prediction_data: Raw prediction response from API
        
    Returns:
        Formatted DataFrame ready for display
    """
    if not prediction_data.get("success") or not prediction_data.get("predictions"):
        return pd.DataFrame()
    
    predictions = prediction_data["predictions"]
    df = pd.DataFrame(predictions)
    
    # Convert timestamp back to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add formatted columns for better display
    df['fraud_status'] = df['fraud_flag'].map({0: 'Normal', 1: 'Suspicious'})
    df['risk_level_display'] = df['risk_level'].str.title()
    df['prediction_score_display'] = df['prediction_score'].round(4)
    
    # Reorder columns for better display
    display_columns = [
        'user_id', 'amount', 'timestamp', 'location', 'device_id',
        'fraud_status', 'risk_level_display', 'prediction_score_display'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in display_columns if col in df.columns]
    df = df[available_columns]
    
    # Rename columns for display
    df = df.rename(columns={
        'risk_level_display': 'Risk Level',
        'prediction_score_display': 'Prediction Score',
        'fraud_status': 'Status'
    })
    
    return df

def get_summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary metrics from prediction results.
    
    Args:
        df: DataFrame with prediction results
        
    Returns:
        Dictionary with summary metrics
    """
    if df.empty:
        return {
            "total_transactions": 0,
            "suspicious_count": 0,
            "normal_count": 0,
            "fraud_percentage": 0.0,
            "avg_prediction_score": 0.0,
            "high_risk_count": 0,
            "low_risk_count": 0
        }
    
    total = len(df)
    suspicious = len(df[df['Status'] == 'Suspicious']) if 'Status' in df.columns else 0
    normal = total - suspicious
    fraud_percentage = (suspicious / total * 100) if total > 0 else 0
    
    # Calculate average prediction score
    score_col = 'Prediction Score' if 'Prediction Score' in df.columns else 'prediction_score'
    avg_score = df[score_col].mean() if score_col in df.columns else 0
    
    # Risk level counts
    risk_col = 'Risk Level' if 'Risk Level' in df.columns else 'risk_level'
    high_risk = len(df[df[risk_col] == 'High']) if risk_col in df.columns else 0
    low_risk = total - high_risk
    
    return {
        "total_transactions": total,
        "suspicious_count": suspicious,
        "normal_count": normal,
        "fraud_percentage": round(fraud_percentage, 2),
        "avg_prediction_score": round(avg_score, 4),
        "high_risk_count": high_risk,
        "low_risk_count": low_risk
    }

def style_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply conditional styling to the DataFrame for better visualization.
    
    Args:
        df: DataFrame to style
        
    Returns:
        Styled DataFrame
    """
    if df.empty or 'Status' not in df.columns:
        return df
    
    def highlight_suspicious(row):
        """Highlight suspicious transactions in red."""
        if row['Status'] == 'Suspicious':
            return ['background-color: #ffebee; color: #c62828'] * len(row)
        elif row['Status'] == 'Normal':
            return ['background-color: #e8f5e8; color: #2e7d32'] * len(row)
        else:
            return [''] * len(row)
    
    return df.style.apply(highlight_suspicious, axis=1)

def convert_timestamp_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp columns for better display in Streamlit.
    
    Args:
        df: DataFrame with timestamp column
        
    Returns:
        DataFrame with formatted timestamp
    """
    df_display = df.copy()
    
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df_display
