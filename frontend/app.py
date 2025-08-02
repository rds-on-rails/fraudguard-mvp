"""
Streamlit frontend for FraudGuard MVP.
Provides a web interface for fraud detection analysis.
"""
import streamlit as st
import pandas as pd
import requests
import json
import sys
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_simulator import generate_transaction_data

# Page configuration
st.set_page_config(
    page_title="FraudGuard MVP",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def call_fraud_api(transactions_data):
    """Call the fraud detection API."""
    try:
        # Prepare the data for API call
        transactions_list = []
        for _, row in transactions_data.iterrows():
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
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to the fraud detection API. Please ensure the backend is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def check_api_health():
    """Check if the API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🛡️ FraudGuard MVP")
    st.markdown("**Real-time fraud detection for financial transactions**")
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Check API health
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API is healthy")
    else:
        st.sidebar.error("❌ API is not available")
        st.sidebar.markdown("Please start the backend server:")
        st.sidebar.code("cd backend && python main.py")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Generate Random Data"]
    )
    
    # Initialize session state
    if 'transaction_data' not in st.session_state:
        st.session_state.transaction_data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Data loading section
    st.header("📊 Data Loading")
    
    if data_source == "Upload CSV File":
        uploaded_file = st.file_uploader(
            "Upload a CSV file with transaction data",
            type=['csv'],
            help="CSV should contain columns: user_id, amount, timestamp, location, device_id"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate required columns
                required_columns = ['user_id', 'amount', 'timestamp', 'location', 'device_id']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.info("Please ensure your CSV has these columns: user_id, amount, timestamp, location, device_id")
                else:
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.session_state.transaction_data = df
                    st.success(f"✅ Loaded {len(df)} transactions from CSV")
                    
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
    
    else:  # Generate Random Data
        col1, col2 = st.columns(2)
        
        with col1:
            num_transactions = st.number_input(
                "Number of transactions to generate",
                min_value=10,
                max_value=10000,
                value=1000,
                step=100
            )
        
        with col2:
            if st.button("🎲 Generate Random Data", type="primary"):
                with st.spinner("Generating transaction data..."):
                    df = generate_transaction_data(num_transactions)
                    st.session_state.transaction_data = df
                    st.success(f"✅ Generated {len(df)} random transactions")
    
    # Display data and run fraud detection
    if st.session_state.transaction_data is not None:
        df = st.session_state.transaction_data
        
        # Fraud Detection Section
        st.header("🔍 Fraud Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Run Fraud Detection", type="primary", disabled=not api_healthy):
                if api_healthy:
                    with st.spinner("Analyzing transactions for fraud..."):
                        result = call_fraud_api(df)
                        
                        if result:
                            st.session_state.predictions = result
                            st.success("✅ Fraud detection completed!")
                        else:
                            st.error("❌ Failed to get fraud predictions")
                else:
                    st.error("API is not available. Please start the backend server.")
        
        with col2:
            if st.session_state.predictions:
                summary = st.session_state.predictions['summary']
                st.metric(
                    "Fraud Detection Rate",
                    f"{summary['fraud_percentage']:.1f}%",
                    f"{summary['fraudulent_transactions']} of {summary['total_transactions']} transactions"
                )
        
        # Results Section
        if st.session_state.predictions:
            st.header("📈 Results")
            
            # Summary metrics
            summary = st.session_state.predictions['summary']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", summary['total_transactions'])
            
            with col2:
                st.metric("Suspicious Transactions", summary['fraudulent_transactions'])
            
            with col3:
                st.metric("Normal Transactions", summary['normal_transactions'])
            
            with col4:
                st.metric("Fraud Rate", f"{summary['fraud_percentage']:.1f}%")
            
            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(st.session_state.predictions['predictions'])
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
            
            # Filters
            st.subheader("🔍 Filters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                show_fraud_only = st.checkbox("Show only suspicious transactions")
            
            with col2:
                fraud_filter = st.selectbox(
                    "Filter by fraud status:",
                    ["All", "Fraudulent Only", "Normal Only"]
                )
            
            # Apply filters
            filtered_df = predictions_df.copy()
            
            if show_fraud_only or fraud_filter == "Fraudulent Only":
                filtered_df = filtered_df[filtered_df['fraud_flag'] == 1]
            elif fraud_filter == "Normal Only":
                filtered_df = filtered_df[filtered_df['fraud_flag'] == 0]
            
            # Data table
            st.subheader("📋 Transaction Details")
            
            if len(filtered_df) > 0:
                # Style the dataframe
                def highlight_fraud(row):
                    if row['fraud_flag'] == 1:
                        return ['background-color: #ffebee'] * len(row)
                    else:
                        return [''] * len(row)
                
                styled_df = filtered_df.style.apply(highlight_fraud, axis=1)
                st.dataframe(styled_df, use_container_width=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No transactions match the current filters.")
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            if len(predictions_df) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fraud distribution pie chart
                    fraud_counts = predictions_df['fraud_flag'].value_counts()
                    fig_pie = px.pie(
                        values=fraud_counts.values,
                        names=['Normal', 'Fraudulent'],
                        title="Transaction Distribution",
                        color_discrete_map={'Normal': '#4CAF50', 'Fraudulent': '#F44336'}
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Amount distribution by fraud status
                    fig_box = px.box(
                        predictions_df,
                        x='fraud_flag',
                        y='amount',
                        title="Amount Distribution by Fraud Status",
                        labels={'fraud_flag': 'Fraud Flag (0=Normal, 1=Fraud)', 'amount': 'Transaction Amount'},
                        color='fraud_flag',
                        color_discrete_map={0: '#4CAF50', 1: '#F44336'}
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Time series plot
                if len(predictions_df) > 1:
                    fig_time = px.scatter(
                        predictions_df.sort_values('timestamp'),
                        x='timestamp',
                        y='amount',
                        color='fraud_flag',
                        title="Transactions Over Time",
                        labels={'fraud_flag': 'Fraud Flag', 'amount': 'Amount'},
                        color_discrete_map={0: '#4CAF50', 1: '#F44336'}
                    )
                    fig_time.update_layout(showlegend=True)
                    st.plotly_chart(fig_time, use_container_width=True)
        
        else:
            # Show sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.info("👆 Run fraud detection to see analysis results")
    
    # Footer
    st.markdown("---")
    st.markdown("**FraudGuard MVP** - Built with Streamlit, FastAPI, and scikit-learn")

if __name__ == "__main__":
    main()
