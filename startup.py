#!/usr/bin/env python3
"""
Azure App Service startup script for FraudGuard Streamlit application
"""
import os
import subprocess
import sys

def main():
    """Start the Streamlit application with Azure-compatible settings"""
    
    # Get the port from environment variable (Azure App Service sets this)
    port = os.environ.get('PORT', '8000')
    
    # Streamlit command with Azure-compatible settings
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"Starting FraudGuard on port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    # Start the application
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
