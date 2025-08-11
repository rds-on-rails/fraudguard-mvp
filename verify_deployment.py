#!/usr/bin/env python3
"""
FraudGuard Azure Deployment Verification Script
Checks if the deployed app is running correctly
"""

import requests
import time
import sys
from urllib.parse import urljoin

def check_app_health(base_url, timeout=30):
    """Check if the Streamlit app is responding"""
    
    print(f"🔍 Checking app health at: {base_url}")
    
    try:
        # Check main page
        response = requests.get(base_url, timeout=timeout)
        
        if response.status_code == 200:
            print("✅ App is responding successfully!")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Size: {len(response.content)} bytes")
            
            # Check if it's actually Streamlit content
            if "streamlit" in response.text.lower() or "fraudguard" in response.text.lower():
                print("✅ Streamlit app detected in response!")
                return True
            else:
                print("⚠️  Response received but may not be Streamlit app")
                return False
                
        else:
            print(f"❌ App returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out after {timeout} seconds")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - app may not be deployed yet")
        return False
    except Exception as e:
        print(f"❌ Error checking app: {str(e)}")
        return False

def main():
    """Main verification function"""
    
    print("🚀 FraudGuard Azure Deployment Verification")
    print("=" * 50)
    
    # Get app URL from user
    if len(sys.argv) > 1:
        app_url = sys.argv[1]
    else:
        app_url = input("Enter your Azure App Service URL (e.g., https://fraudguard-app.azurewebsites.net): ")
    
    # Ensure URL has proper format
    if not app_url.startswith('http'):
        app_url = f"https://{app_url}"
    
    print(f"🌐 Target URL: {app_url}")
    print()
    
    # Initial check
    if check_app_health(app_url):
        print("\n🎉 Deployment verification successful!")
        print("\n📋 Next steps:")
        print("   1. Open the app in your browser")
        print("   2. Test the fraud detection functionality")
        print("   3. Try generating sample data")
        print("   4. Verify CSV export works")
    else:
        print("\n⚠️  App may still be starting up...")
        print("   - Azure deployments can take 5-10 minutes")
        print("   - Check Azure Portal logs if issues persist")
        print("   - Verify all configuration settings")
    
    print(f"\n🔗 Direct link: {app_url}")

if __name__ == "__main__":
    main()
