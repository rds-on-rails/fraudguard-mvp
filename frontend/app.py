"""
Enhanced Interactive Streamlit Frontend for FraudGuard MVP.
Provides live fraud detection with visual displays and interactive filtering.
"""
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page configuration
st.set_page_config(
    page_title="FraudGuard MVP - Live Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

def generate_sample_data(n_transactions=50):
    """Generate sample transaction data for testing."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/generate-data",
            params={"n": n_transactions},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['transactions'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            st.error(f"Failed to generate data: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        return None

def call_fraud_prediction_api(transactions_df):
    """Call the fraud detection API with DataFrame."""
    try:
        # Prepare the data for API call
        transactions_list = []
        for _, row in transactions_df.iterrows():
            transaction = {
                "transaction_id": str(row.get('transaction_id', f"tx_{np.random.randint(1000, 9999)}")),
                "amount": float(row['amount']),
                "timestamp": row['timestamp'].isoformat() if pd.notnull(row['timestamp']) else datetime.now().isoformat(),
                "location": str(row['location']),
                "device_type": str(row.get('device_type', 'mobile')),
                "merchant_id": str(row.get('merchant_id', 'merchant_001')),
                "user_id": str(row['user_id'])
            }
            transactions_list.append(transaction)
        
        payload = {"transactions": transactions_list}
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return pd.DataFrame(result['predictions'])
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
    """Enhanced Interactive Streamlit Application."""
    
    # Header
    st.title("🛡️ FraudGuard MVP - Live Fraud Detection")
    st.markdown("**Interactive real-time fraud detection with ML-powered insights**")
    
    # Initialize session state
    if 'transactions_df' not in st.session_state:
        st.session_state.transactions_df = None
    if 'predictions_df' not in st.session_state:
        st.session_state.predictions_df = None
    
    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Health Check
    api_healthy = check_api_health()
    if api_healthy:
        st.sidebar.success("✅ API is healthy")
    else:
        st.sidebar.error("❌ API is not available")
        st.sidebar.markdown("Please start the backend server:")
        st.sidebar.code("cd backend && python main.py")
        st.stop()
    
    # 🚀 Live Prediction Feature
    st.header("🚀 Live Fraud Detection")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        n_transactions = st.slider("Number of transactions to generate:", 10, 200, 50)
    
    with col2:
        if st.button("🎲 Generate & Detect Fraud", type="primary"):
            with st.spinner("Generating transactions and detecting fraud..."):
                # Generate sample data
                transactions_df = generate_sample_data(n_transactions)
                if transactions_df is not None:
                    st.session_state.transactions_df = transactions_df
                    
                    # Get fraud predictions
                    predictions_df = call_fraud_prediction_api(transactions_df)
                    if predictions_df is not None:
                        st.session_state.predictions_df = predictions_df
                        st.success(f"✅ Processed {len(predictions_df)} transactions!")
    
    with col3:
        if st.button("🗑️ Clear Results"):
            st.session_state.transactions_df = None
            st.session_state.predictions_df = None
            st.rerun()
    
    # Display results if available
    if st.session_state.predictions_df is not None:
        predictions_df = st.session_state.predictions_df.copy()
        
        # Convert timestamp column for filtering
        if 'timestamp' in predictions_df.columns:
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # 📊 Sidebar Filters
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        if 'timestamp' in predictions_df.columns:
            min_date = predictions_df['timestamp'].min().date()
            max_date = predictions_df['timestamp'].max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range:",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
        
        # Amount range filter
        min_amount = float(predictions_df['amount'].min())
        max_amount = float(predictions_df['amount'].max())
        
        amount_range = st.sidebar.slider(
            "Amount Range ($):",
            min_value=min_amount,
            max_value=max_amount,
            value=(min_amount, max_amount),
            step=1.0
        )
        
        # Fraud filter toggle
        fraud_filter = st.sidebar.selectbox(
            "Show Transactions:",
            options=["All", "Fraudulent Only", "Normal Only"]
        )
        
        # Apply filters
        filtered_df = predictions_df.copy()
        
        # Date filter
        if 'timestamp' in filtered_df.columns and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) &
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        
        # Amount filter
        filtered_df = filtered_df[
            (filtered_df['amount'] >= amount_range[0]) &
            (filtered_df['amount'] <= amount_range[1])
        ]
        
        # Fraud filter
        if fraud_filter == "Fraudulent Only":
            filtered_df = filtered_df[filtered_df['fraud_flag'] == 1]
        elif fraud_filter == "Normal Only":
            filtered_df = filtered_df[filtered_df['fraud_flag'] == 0]
        
        # 📈 Metrics Dashboard
        st.header("📈 Metrics Dashboard")
        
        total_transactions = len(filtered_df)
        total_frauds = len(filtered_df[filtered_df['fraud_flag'] == 1])
        fraud_percentage = (total_frauds / total_transactions * 100) if total_transactions > 0 else 0
        total_amount = filtered_df['amount'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{total_transactions:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Fraudulent Transactions", 
                f"{total_frauds:,}",
                delta=f"{fraud_percentage:.1f}%"
            )
        
        with col3:
            st.metric(
                "Fraud Rate",
                f"{fraud_percentage:.1f}%",
                delta=None
            )
        
        with col4:
            st.metric(
                "Total Amount",
                f"${total_amount:,.2f}",
                delta=None
            )
        
        # Fraud percentage gauge chart
        if total_transactions > 0:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = fraud_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Percentage"},
                delta = {'reference': 5.0},  # Reference fraud rate
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 5], 'color': "lightgray"},
                        {'range': [5, 15], 'color': "yellow"},
                        {'range': [15, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 📊 Visual Display with Color Coding
        st.header("📊 Transaction Analysis")
        
        if total_transactions > 0:
            # Sortable DataFrame with Color Coding
            st.subheader("📋 Transaction Details")
            
            # Sorting options
            sort_options = ['timestamp', 'amount', 'fraud_flag', 'user_id']
            sort_by = st.selectbox("Sort by:", sort_options, index=0)
            sort_ascending = st.checkbox("Ascending order", value=False)
            
            # Sort the dataframe
            display_df = filtered_df.sort_values(by=sort_by, ascending=sort_ascending)
            
            # Color coding function
            def highlight_fraud_rows(row):
                if row['fraud_flag'] == 1:
                    return ['background-color: #ffcdd2; color: #d32f2f'] * len(row)  # Red for fraud
                else:
                    return ['background-color: #c8e6c9; color: #388e3c'] * len(row)  # Green for normal
            
            # Apply styling and display
            styled_df = display_df.style.apply(highlight_fraud_rows, axis=1)
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # CSV Download Feature
            st.subheader("📥 Export Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Download all results
                csv_all = display_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download All Results",
                    data=csv_all,
                    file_name=f"fraud_detection_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download only flagged results
                flagged_df = display_df[display_df['fraud_flag'] == 1]
                if len(flagged_df) > 0:
                    csv_flagged = flagged_df.to_csv(index=False)
                    st.download_button(
                        label="🚨 Download Flagged Only",
                        data=csv_flagged,
                        file_name=f"fraud_detection_flagged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("No fraudulent transactions to download")
            
            # 📊 Interactive Visualizations
            st.subheader("📊 Interactive Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Fraud Distribution Pie Chart
                fraud_counts = display_df['fraud_flag'].value_counts()
                labels = ['Normal', 'Fraudulent']
                values = [fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
                
                fig_pie = px.pie(
                    values=values,
                    names=labels,
                    title="Transaction Distribution",
                    color_discrete_map={'Normal': '#4CAF50', 'Fraudulent': '#F44336'}
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with viz_col2:
                # Amount Distribution Box Plot
                fig_box = px.box(
                    display_df,
                    x='fraud_flag',
                    y='amount',
                    title="Amount Distribution by Fraud Status",
                    labels={'fraud_flag': 'Fraud Flag (0=Normal, 1=Fraud)', 'amount': 'Amount ($)'},
                    color='fraud_flag',
                    color_discrete_map={0: '#4CAF50', 1: '#F44336'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Time Series Scatter Plot
            if 'timestamp' in display_df.columns and len(display_df) > 1:
                fig_time = px.scatter(
                    display_df.sort_values('timestamp'),
                    x='timestamp',
                    y='amount',
                    color='fraud_flag',
                    title="Transactions Over Time",
                    labels={'fraud_flag': 'Fraud Status', 'amount': 'Amount ($)'},
                    color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                    hover_data=['user_id', 'location']
                )
                fig_time.update_layout(showlegend=True)
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Amount Histogram
            fig_hist = px.histogram(
                display_df,
                x='amount',
                color='fraud_flag',
                title="Amount Distribution Histogram",
                labels={'amount': 'Amount ($)', 'count': 'Number of Transactions'},
                color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        else:
            st.info("No transactions match the current filters. Try adjusting your filter settings.")
    
    else:
        # No data loaded yet
        st.info("👆 Click 'Generate & Detect Fraud' to start analyzing transactions!")
        
        # Show example of what the system can do
        st.header("🎯 What This System Can Do")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🤖 AI-Powered Detection**
            - Uses Isolation Forest ML algorithm
            - Trained on realistic transaction patterns
            - Detects anomalies in real-time
            """)
        
        with col2:
            st.markdown("""
            **📊 Rich Analytics**
            - Interactive visualizations
            - Sortable and filterable data
            - Real-time metrics dashboard
            """)
        
        with col3:
            st.markdown("""
            **💾 Export Capabilities**
            - Download all results as CSV
            - Export only flagged transactions
            - Time-stamped file names
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**FraudGuard MVP** - Enhanced Interactive Fraud Detection with ML-powered Analytics")
    st.markdown("Built with Streamlit, FastAPI, scikit-learn, and Plotly")

if __name__ == "__main__":
    main()
