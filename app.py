"""
🛡️ FraudGuard MVP - Unified Streamlit App for Cloud Deployment
Real-Time Transaction Anomaly Detection System

This unified app combines all ML functionality directly in Streamlit
for seamless cloud deployment without needing a separate backend.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import uuid
from faker import Faker
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
import io
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Faker for data generation
fake = Faker()

# Page configuration
st.set_page_config(
    page_title="🛡️ FraudGuard MVP", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .suspicious-row {
        background-color: #ffebee !important;
        border: 1px solid #f44336 !important;
    }
    .normal-row {
        background-color: #e8f5e8 !important;
        border: 1px solid #4caf50 !important;
    }
    .status-suspicious {
        color: #d32f2f;
        font-weight: bold;
    }
    .status-normal {
        color: #388e3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class FraudDetectionModel:
    """
    Unified fraud detection model for Streamlit Cloud deployment.
    Combines all ML functionality in a single class.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
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
        self.training_info = {}
        
    def generate_dummy_data(self, n: int = 1000) -> pd.DataFrame:
        """Generate synthetic transaction data for training and testing."""
        logger.info(f"Generating {n} dummy transactions")
        
        transactions = []
        for _ in range(n):
            transaction = {
                'user_id': str(uuid.uuid4()),
                'amount': fake.pyfloat(left_digits=5, right_digits=2, positive=True, min_value=1.0, max_value=10000.0),
                'timestamp': fake.date_time_between(start_date='-30d', end_date='now'),
                'location': fake.city(),
                'device_id': str(uuid.uuid4())
            }
            transactions.append(transaction)
            
        df = pd.DataFrame(transactions)
        logger.info(f"Generated {len(df)} transactions")
        return df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from transaction data for ML model."""
        features_df = df.copy()
        
        # Convert timestamp to numeric features
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # Amount-based features
        features_df['log_amount'] = np.log1p(features_df['amount'])
        
        # Location encoding (simple hash-based)
        features_df['location_encoded'] = features_df['location'].astype(str).apply(
            lambda x: hash(x) % 1000
        )
        
        # User and device behavior features
        user_stats = features_df.groupby('user_id')['amount'].agg(['count', 'mean', 'std']).fillna(0)
        user_stats.columns = ['user_transaction_count', 'user_avg_amount', 'user_amount_std']
        features_df = features_df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        
        device_stats = features_df.groupby('device_id')['amount'].agg(['count', 'mean']).fillna(0)
        device_stats.columns = ['device_transaction_count', 'device_avg_amount']
        features_df = features_df.merge(device_stats, left_on='device_id', right_index=True, how='left')
        
        # Select feature columns for ML model
        feature_cols = [
            'amount', 'log_amount', 'hour', 'day_of_week', 'is_weekend',
            'location_encoded', 'user_transaction_count', 'user_avg_amount', 
            'user_amount_std', 'device_transaction_count', 'device_avg_amount'
        ]
        
        return features_df[feature_cols].fillna(0)
    
    def train_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the fraud detection model."""
        logger.info(f"Training model with {len(data)} transactions")
        
        # Extract features
        features = self.extract_features(data)
        self.feature_columns = features.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Store training info
        self.training_info = {
            'training_samples': len(data),
            'feature_count': len(self.feature_columns),
            'contamination_rate': self.contamination,
            'model_type': 'Isolation Forest',
            'trained_at': datetime.now().isoformat()
        }
        
        logger.info("Model training completed successfully")
        return self.training_info
    
    def predict_fraud(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Predict fraud for transaction data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_features(data)
        
        # Ensure feature consistency
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Get predictions and scores
        predictions = self.model.predict(X_scaled)  # 1 for normal, -1 for anomaly
        scores = self.model.decision_function(X_scaled)  # Anomaly scores
        
        # Process results
        results = []
        for i, (pred, score) in enumerate(zip(predictions, scores)):
            fraud_flag = 1 if pred == -1 else 0
            risk_level = "High Risk" if fraud_flag == 1 else "Low Risk"
            
            results.append({
                'fraud_flag': fraud_flag,
                'prediction_score': float(score),
                'risk_level': risk_level,
                'status': 'Suspicious' if fraud_flag == 1 else 'Normal'
            })
        
        return results

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = FraudDetectionModel()
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ FraudGuard MVP</h1>
        <p>Real-Time Transaction Anomaly Detection System</p>
        <p><em>Powered by Machine Learning • Built with Streamlit</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model info and controls
    with st.sidebar:
        st.header("🔧 Model Information")
        
        if st.session_state.model.is_trained:
            info = st.session_state.model.training_info
            st.success("✅ Model Trained")
            st.write(f"**Training Samples:** {info['training_samples']:,}")
            st.write(f"**Features:** {info['feature_count']}")
            st.write(f"**Contamination Rate:** {info['contamination_rate']}%")
            st.write(f"**Model Type:** {info['model_type']}")
        else:
            st.warning("⚠️ Model Not Trained")
            st.write("Generate sample data to train the model")
        
        st.header("📊 Quick Actions")
        
        # Train model button
        if st.button("🎯 Auto-Train Model", use_container_width=True):
            with st.spinner("Training model..."):
                training_data = st.session_state.model.generate_dummy_data(1000)
                st.session_state.model.train_model(training_data)
                st.success("Model trained successfully!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Data Input & Analysis")
        
        # Data input options
        input_method = st.radio(
            "Choose input method:",
            ["Generate Sample Data", "Upload CSV File"],
            horizontal=True
        )
        
        if input_method == "Generate Sample Data":
            col_gen1, col_gen2 = st.columns([3, 1])
            with col_gen1:
                num_samples = st.slider("Number of samples", 10, 1000, 100)
            with col_gen2:
                if st.button("🎲 Generate Data", use_container_width=True):
                    with st.spinner("Generating sample data..."):
                        st.session_state.current_data = st.session_state.model.generate_dummy_data(num_samples)
                        st.success(f"Generated {num_samples} transactions!")
        
        else:  # Upload CSV
            uploaded_file = st.file_uploader(
                "Upload transaction CSV file", 
                type=['csv'],
                help="CSV should contain: user_id, amount, timestamp, location, device_id"
            )
            
            if uploaded_file is not None:
                try:
                    st.session_state.current_data = pd.read_csv(uploaded_file)
                    
                    # Validate required columns
                    required_cols = ['user_id', 'amount', 'timestamp', 'location', 'device_id']
                    missing_cols = [col for col in required_cols if col not in st.session_state.current_data.columns]
                    
                    if missing_cols:
                        st.error(f"Missing columns: {', '.join(missing_cols)}")
                        st.session_state.current_data = None
                    else:
                        st.success(f"Uploaded {len(st.session_state.current_data)} transactions!")
                
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    st.session_state.current_data = None
    
    with col2:
        st.header("🎯 Model Actions")
        
        # Prediction button
        if st.session_state.current_data is not None and st.session_state.model.is_trained:
            if st.button("🔍 Run Fraud Detection", use_container_width=True, type="primary"):
                with st.spinner("Analyzing transactions..."):
                    predictions = st.session_state.model.predict_fraud(st.session_state.current_data)
                    st.session_state.predictions = predictions
                    st.success("Analysis complete!")
                    st.rerun()
        else:
            if st.session_state.current_data is None:
                st.info("📝 Load data first")
            if not st.session_state.model.is_trained:
                st.info("🎯 Train model first")
    
    # Display results
    if st.session_state.current_data is not None:
        st.header("📊 Transaction Data & Results")
        
        # Create display dataframe
        display_df = st.session_state.current_data.copy()
        
        if st.session_state.predictions is not None:
            # Add prediction results
            pred_df = pd.DataFrame(st.session_state.predictions)
            display_df = pd.concat([display_df, pred_df], axis=1)
            
            # Summary metrics
            total_transactions = len(display_df)
            suspicious_count = len(display_df[display_df['fraud_flag'] == 1])
            fraud_percentage = (suspicious_count / total_transactions) * 100 if total_transactions > 0 else 0
            
            # Metrics display
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Total Transactions", f"{total_transactions:,}")
            with col_m2:
                st.metric("Suspicious", f"{suspicious_count:,}", delta=f"{fraud_percentage:.1f}%")
            with col_m3:
                normal_count = total_transactions - suspicious_count
                st.metric("Normal", f"{normal_count:,}")
            with col_m4:
                if suspicious_count > 0:
                    avg_fraud_score = display_df[display_df['fraud_flag'] == 1]['prediction_score'].mean()
                    st.metric("Avg Fraud Score", f"{avg_fraud_score:.3f}")
                else:
                    st.metric("Avg Fraud Score", "N/A")
        
        # Filtering options
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            show_suspicious_only = st.checkbox("Show only suspicious transactions")
        with col_f2:
            if st.session_state.predictions is not None:
                risk_filter = st.selectbox("Filter by risk level", ["All", "High Risk", "Low Risk"])
            else:
                risk_filter = "All"
        
        # Apply filters
        filtered_df = display_df.copy()
        if st.session_state.predictions is not None:
            if show_suspicious_only:
                filtered_df = filtered_df[filtered_df['fraud_flag'] == 1]
            if risk_filter != "All":
                filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        
        # Display data table
        if len(filtered_df) > 0:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"fraudguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No transactions match the current filters.")
        
        # Visualizations
        if st.session_state.predictions is not None and len(display_df) > 0:
            st.header("📈 Data Visualizations")
            
            col_v1, col_v2 = st.columns(2)
            
            with col_v1:
                # Fraud distribution pie chart
                fraud_counts = display_df['status'].value_counts()
                fig_pie = px.pie(
                    values=fraud_counts.values, 
                    names=fraud_counts.index,
                    title="Transaction Status Distribution",
                    color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_v2:
                # Amount distribution by status
                fig_box = px.box(
                    display_df, 
                    x='status', 
                    y='amount',
                    title="Amount Distribution by Status",
                    color='status',
                    color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Prediction score histogram
            fig_hist = px.histogram(
                display_df, 
                x='prediction_score',
                color='status',
                title="Prediction Score Distribution",
                nbins=30,
                color_discrete_map={'Suspicious': '#ff4444', 'Normal': '#44ff44'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()
