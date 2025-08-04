"""Enhanced FastAPI backend for FraudGuard MVP.
Provides fraud detection API endpoints with improved error handling and validation.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
from fastapi import Query
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import sys
import os
import traceback

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import generate_dummy_data, train_model, predict_fraud, get_model_info, is_model_trained
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard API Enhanced",
    description="Advanced fraud detection API using Isolation Forest with comprehensive error handling",
    version="2.0.0"
)

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid input data",
            "detail": str(exc),
            "type": "validation_error"
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "type": "server_error"
        }
    )

class Transaction(BaseModel):
    """Enhanced transaction data model with validation."""
    user_id: str = Field(..., min_length=1, description="User identifier (UUID recommended)")
    amount: float = Field(..., gt=0, le=1000000, description="Transaction amount (must be positive)")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    location: str = Field(..., min_length=1, max_length=100, description="Transaction location")
    device_id: str = Field(..., min_length=1, description="Device identifier")
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:
            raise ValueError('Amount cannot exceed 1,000,000')
        return round(v, 2)
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        if v > datetime.now():
            raise ValueError('Timestamp cannot be in the future')
        # Allow timestamps up to 1 year in the past
        if v < datetime.now().replace(year=datetime.now().year - 1):
            raise ValueError('Timestamp cannot be more than 1 year old')
        return v

class TransactionBatch(BaseModel):
    """Batch of transactions for prediction with validation."""
    transactions: List[Transaction] = Field(..., min_items=1, max_items=1000, 
                                            description="List of transactions (1-1000 items)")

class PredictionResult(BaseModel):
    """Individual prediction result."""
    user_id: str
    amount: float
    timestamp: datetime
    location: str
    device_id: str
    fraud_flag: int = Field(..., description="0 = normal, 1 = suspicious")
    prediction_score: float = Field(..., description="Model confidence score")
    risk_level: str = Field(..., description="Risk level: low or high")

class PredictionResponse(BaseModel):
    """Enhanced response model for fraud predictions."""
    success: bool
    predictions: List[PredictionResult]
    summary: Dict[str, Any]
    model_info: Dict[str, Any]
    timestamp: str

class ErrorResponse(BaseModel):
    """Standardized error response."""
    success: bool = False
    error: str
    detail: str
    type: str
    timestamp: str

@app.on_event("startup")
async def startup_event():
    """Initialize and train the model on startup."""
    try:
        logger.info("🚀 Starting FraudGuard Enhanced API...")
        logger.info("📊 Initializing fraud detection model...")
        
        # Train model (generates its own training data)
        model_info = train_model()
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"   - Training samples: {model_info['training_samples']}")
        logger.info(f"   - Feature count: {model_info['feature_count']}")
        logger.info(f"   - Anomaly detection rate: {model_info['anomaly_rate']}%")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize fraud detection model: {e}")
        logger.error("⚠️  API will start but fraud detection will not be available")

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FraudGuard Enhanced API",
        "version": "2.0.0",
        "status": "healthy",
        "model_trained": is_model_trained(),
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "generate_data": "/generate-data",
            "docs": "/docs"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with model status."""
    model_info = get_model_info()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": model_info,
        "api_version": "2.0.0",
        "uptime_check": "passed"
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Fraud Detection"])
async def predict_fraud_endpoint(batch: TransactionBatch):
    """
    Predict fraud for a batch of transactions with comprehensive error handling.
    
    Args:
        batch: Batch of 1-1000 transactions to analyze
        
    Returns:
        Detailed predictions with fraud flags, scores, and summary statistics
    
    Raises:
        HTTPException: For various error conditions with specific error codes
    """
    try:
        # Check if model is trained
        if not is_model_trained():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not trained. Please wait for model initialization or contact support."
            )
        
        logger.info(f"🔍 Processing {len(batch.transactions)} transactions for fraud detection")
        
        # Convert to DataFrame
        transactions_data = [tx.dict() for tx in batch.transactions]
        df = pd.DataFrame(transactions_data)
        
        # Get predictions
        try:
            result_df = predict_fraud(df)
            predictions = result_df.to_dict('records')
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Prediction failed due to invalid data: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Fraud prediction service temporarily unavailable"
            )
        
        # Calculate summary statistics
        total_transactions = len(predictions)
        fraudulent_count = sum(1 for pred in predictions if pred['fraud_flag'] == 1)
        normal_count = total_transactions - fraudulent_count
        fraud_percentage = (fraudulent_count / total_transactions) * 100 if total_transactions > 0 else 0
        
        # Calculate risk distribution
        high_risk_count = fraudulent_count
        low_risk_count = normal_count
        
        # Average prediction scores
        fraud_scores = [p['prediction_score'] for p in predictions if p['fraud_flag'] == 1]
        normal_scores = [p['prediction_score'] for p in predictions if p['fraud_flag'] == 0]
        
        summary = {
            "total_transactions": total_transactions,
            "fraudulent_transactions": fraudulent_count,
            "normal_transactions": normal_count,
            "fraud_percentage": round(fraud_percentage, 2),
            "risk_distribution": {
                "high_risk": high_risk_count,
                "low_risk": low_risk_count
            },
            "score_statistics": {
                "avg_fraud_score": round(sum(fraud_scores) / len(fraud_scores), 4) if fraud_scores else 0,
                "avg_normal_score": round(sum(normal_scores) / len(normal_scores), 4) if normal_scores else 0
            }
        }
        
        # Get current model info
        model_info = get_model_info()
        
        logger.info(f"✅ Fraud detection completed: {fraudulent_count}/{total_transactions} flagged as fraudulent ({fraud_percentage:.1f}%)")
        
        return PredictionResponse(
            success=True,
            predictions=predictions,
            summary=summary,
            model_info=model_info,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"❌ Unexpected error during fraud prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during fraud detection"
        )

@app.get("/model/info", tags=["Model"])
async def get_model_information():
    """Get detailed information about the current fraud detection model."""
    try:
        model_info = get_model_info()
        return {
            "success": True,
            "model": model_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )

@app.post("/generate-data", tags=["Data Generation"])
async def generate_test_data(n: int = Query(100, ge=10, le=1000, description="Number of records to generate")):
    """Generate dummy transaction data for testing purposes."""
    try:
        logger.info(f"Generating {n} test transaction records...")
        
        data = generate_dummy_data(n=n)
        transactions = data.to_dict('records')
        
        # Convert timestamps to ISO format for JSON serialization
        for tx in transactions:
            tx['timestamp'] = tx['timestamp'].isoformat()
        
        return {
            "success": True,
            "count": len(transactions),
            "transactions": transactions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating test data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate test data"
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting FraudGuard Enhanced API server...")
    
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
