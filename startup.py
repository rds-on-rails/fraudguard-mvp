#!/usr/bin/env python3
"""
Azure App Service startup script for FraudGuard Streamlit application
Optimized for Python 3.9 and Azure App Service
"""
import os
import subprocess
import sys
import logging

# Configure logging for Azure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Start the Streamlit application with Azure-compatible settings"""
    
    # Get the port from environment variable (Azure App Service sets this)
    port = os.environ.get('PORT', '8000')
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Starting FraudGuard on port {port}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        logger.error("app.py not found in current directory")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        sys.exit(1)
    
    # Streamlit command with Azure-compatible settings
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.gatherUsageStats', 'false',
        '--server.fileWatcherType', 'none'
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the application
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start Streamlit: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
