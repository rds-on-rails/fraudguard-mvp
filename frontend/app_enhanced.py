"""Enhanced Streamlit frontend for FraudGuard MVP.
Provides a comprehensive web interface for fraud detection analysis with advanced features.
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
import numpy as np

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.frontend_helpers import (
    check_api_health, get_api_info, generate_sample_data, predict_fraud,
    validate_transaction_data, format_prediction_results, get_summary_metrics,
    style_dataframe, convert_timestamp_for_display
)

# Page configuration
st.set_page_config(
    page_title="FraudGuard Enhanced",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
.header-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton > button {
    width: 100%;
    border-radius: 0.5rem;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
    transition: all 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'transaction_data' not in st.session_state:
        st.session_state.transaction_data = None
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'formatted_predictions' not in st.session_state:
        st.session_state.formatted_predictions = None

def display_header():
    """Display the main header section."""
    st.markdown("""
    <div class="header-container">
        <h1>🛡️ FraudGuard – Real-Time Transaction Anomaly Detector</h1>
        <p>Advanced AI-powered fraud detection system with comprehensive analytics</p>
    </div>
    """, unsafe_allow_html=True)

def display_api_status():
    """Display API connection status in sidebar."""
    st.sidebar.header("🔧 System Status")
    
    # Check API health
    api_healthy = check_api_health()
    api_info = get_api_info()
    
    if api_healthy and api_info:
        st.sidebar.success("✅ API Connected")
        st.sidebar.write(f"**Version:** {api_info.get('version', 'Unknown')}")
        st.sidebar.write(f"**Model Trained:** {'✅' if api_info.get('model_trained') else '❌'}")
    else:
        st.sidebar.error("❌ API Unavailable")
        st.sidebar.markdown("""
        **Start the backend server:**
        ```bash
        cd backend
        python main.py
        ```
        """)
    
    return api_healthy

def section_data_input(api_healthy):
    """Section 1: Header and Options for data input."""
    st.header("📊 Section 1: Data Input Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload CSV Option
        st.subheader("📁 Upload CSV")
        uploaded_file = st.file_uploader(
            "Upload transaction data",
            type=['csv'],
            help="Required columns: user_id, amount, timestamp, location, device_id",
            key="csv_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate data
                is_valid, errors = validate_transaction_data(df)
                
                if is_valid:
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.session_state.transaction_data = df
                    st.success(f"✅ Loaded {len(df):,} transactions from CSV")
                    
                    # Show data preview
                    with st.expander("👀 Data Preview", expanded=False):
                        st.dataframe(df.head(), use_container_width=True)
                else:
                    st.error("❌ Data Validation Failed:")
                    for error in errors:
                        st.write(f"• {error}")
                        
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
    
    with col2:
        # Generate Sample Data Option
        st.subheader("🎲 Generate Sample Data")
        
        num_records = st.number_input(
            "Number of records to generate",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            key="sample_size"
        )
        
        if st.button("🚀 Generate Sample Data", type="primary", disabled=not api_healthy):
            if api_healthy:
                with st.spinner("🤖 Generating sample transaction data..."):
                    sample_data = generate_sample_data(n=num_records)
                    
                    if sample_data is not None:
                        st.session_state.transaction_data = sample_data
                        st.success(f"✅ Generated {len(sample_data):,} sample transactions")
                        
                        # Show data preview
                        with st.expander("👀 Generated Data Preview", expanded=True):
                            st.dataframe(sample_data.head(), use_container_width=True)
                    else:
                        st.error("❌ Failed to generate sample data")
            else:
                st.error("❌ API connection required for data generation")

def section_data_display(api_healthy):
    """Section 2: Data Display with prediction functionality."""
    if st.session_state.transaction_data is None:
        st.info("👆 Please upload CSV data or generate sample data to continue")
        return
    
    st.header("🔍 Section 2: Data Display & Fraud Detection")
    
    df = st.session_state.transaction_data
    
    # Display current data info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Transactions", f"{len(df):,}")
    
    with col2:
        st.metric("💰 Total Amount", f"${df['amount'].sum():,.2f}")
    
    with col3:
        st.metric("👥 Unique Users", f"{df['user_id'].nunique():,}")
    
    # Show current data table
    st.subheader("📋 Current Transaction Data")
    display_df = convert_timestamp_for_display(df)
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # Run Prediction Button
    st.subheader("🤖 Run Fraud Detection")
    
    if st.button("🚀 Analyze Transactions for Fraud", type="primary", disabled=not api_healthy):
        if api_healthy:
            with st.spinner("🔍 Analyzing transactions for fraud patterns..."):
                prediction_results = predict_fraud(df)
                
                if prediction_results and prediction_results.get("success"):
                    st.session_state.prediction_results = prediction_results
                    
                    # Format predictions for display
                    formatted_df = format_prediction_results(prediction_results)
                    st.session_state.formatted_predictions = formatted_df
                    
                    st.success("✅ Fraud detection analysis completed!")
                    
                    # Display quick summary
                    summary = prediction_results.get("summary", {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "🚨 Suspicious Transactions",
                            summary.get("fraudulent_transactions", 0),
                            f"{summary.get('fraud_percentage', 0):.1f}% of total"
                        )
                    
                    with col2:
                        st.metric(
                            "✅ Normal Transactions",
                            summary.get("normal_transactions", 0)
                        )
                else:
                    st.error("❌ Fraud detection failed. Please check the API connection.")
        else:
            st.error("❌ API connection required for fraud detection")

def section_filtering_summary():
    """Section 3: Filtering and Summary with advanced analytics."""
    if st.session_state.formatted_predictions is None:
        st.info("👆 Please run fraud detection to see results")
        return
    
    st.header("📈 Section 3: Results Analysis & Filtering")
    
    prediction_results = st.session_state.prediction_results
    formatted_df = st.session_state.formatted_predictions
    
    # Summary Metrics Dashboard
    st.subheader("📊 Summary Dashboard")
    
    summary = prediction_results.get("summary", {})
    
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🔢 Total Transactions",
            f"{summary.get('total_transactions', 0):,}"
        )
    
    with col2:
        st.metric(
            "🚨 Suspicious Count",
            f"{summary.get('fraudulent_transactions', 0):,}",
            f"{summary.get('fraud_percentage', 0):.1f}%"
        )
    
    with col3:
        st.metric(
            "✅ Normal Count",
            f"{summary.get('normal_transactions', 0):,}"
        )
    
    with col4:
        fraud_pct = summary.get('fraud_percentage', 0)
        st.metric(
            "📈 Fraud Rate",
            f"{fraud_pct:.2f}%",
            f"{'🔴 High' if fraud_pct > 10 else '🟡 Medium' if fraud_pct > 5 else '🟢 Low'}"
        )
    
    # Filtering Options
    st.subheader("🔍 Filter & View Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_suspicious_only = st.checkbox(
            "🚨 Show only suspicious transactions",
            key="suspicious_filter"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by status:",
            ["All Transactions", "Suspicious Only", "Normal Only"],
            key="status_filter"
        )
    
    with col3:
        if 'Risk Level' in formatted_df.columns:
            risk_filter = st.selectbox(
                "Filter by risk level:",
                ["All Risk Levels", "High Risk", "Low Risk"],
                key="risk_filter"
            )
        else:
            risk_filter = "All Risk Levels"
    
    # Apply filters
    filtered_df = formatted_df.copy()
    
    if show_suspicious_only or status_filter == "Suspicious Only":
        filtered_df = filtered_df[filtered_df['Status'] == 'Suspicious']
    elif status_filter == "Normal Only":
        filtered_df = filtered_df[filtered_df['Status'] == 'Normal']
    
    if risk_filter == "High Risk" and 'Risk Level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Risk Level'] == 'High']
    elif risk_filter == "Low Risk" and 'Risk Level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Risk Level'] == 'Low']
    
    # Display Results Table
    st.subheader("📋 Detailed Transaction Results")
    
    if len(filtered_df) > 0:
        # Apply conditional formatting
        def highlight_transactions(row):
            if row['Status'] == 'Suspicious':
                return ['background-color: #ffebee; color: #c62828; font-weight: bold'] * len(row)
            elif row['Status'] == 'Normal':
                return ['background-color: #e8f5e8; color: #2e7d32'] * len(row)
            else:
                return [''] * len(row)
        
        # Format display dataframe
        display_filtered = convert_timestamp_for_display(filtered_df)
        styled_df = display_filtered.style.apply(highlight_transactions, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Results (CSV)",
                data=csv,
                file_name=f"fraudguard_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary"
            )
        
        # Visualizations
        st.subheader("📊 Data Visualizations")
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution pie chart
            status_counts = filtered_df['Status'].value_counts()
            if len(status_counts) > 0:
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Transaction Status Distribution",
                    color_discrete_map={'Normal': '#4CAF50', 'Suspicious': '#F44336'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Amount distribution by status
            if len(filtered_df) > 1:
                fig_box = px.box(
                    filtered_df,
                    x='Status',
                    y='amount',
                    title="Amount Distribution by Status",
                    color='Status',
                    color_discrete_map={'Normal': '#4CAF50', 'Suspicious': '#F44336'}
                )
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Prediction Score Distribution
        if 'Prediction Score' in filtered_df.columns:
            fig_hist = px.histogram(
                filtered_df,
                x='Prediction Score',
                color='Status',
                title="Prediction Score Distribution",
                color_discrete_map={'Normal': '#4CAF50', 'Suspicious': '#F44336'},
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
    else:
        st.info("No transactions match the current filters.")

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Check API status
    api_healthy = display_api_status()
    
    # Main sections
    section_data_input(api_healthy)
    st.divider()
    
    section_data_display(api_healthy)
    st.divider()
    
    section_filtering_summary()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p><strong>FraudGuard Enhanced</strong> - Built with Streamlit, FastAPI, and scikit-learn</p>
        <p>Advanced AI-powered fraud detection with comprehensive analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
