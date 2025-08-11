#!/bin/bash

# Azure App Service startup script for Streamlit
echo "Starting FraudGuard Streamlit application..."

# Install dependencies
pip install -r requirements.txt

# Start Streamlit with Azure-compatible settings
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
