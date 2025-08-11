# ✅ FraudGuard Azure Deployment Checklist

## Pre-Deployment (Completed)
- [x] Azure configuration files created
- [x] Code committed and pushed to GitHub
- [x] Streamlit app optimized for cloud deployment
- [x] Requirements.txt configured
- [x] Startup scripts prepared

## Azure Portal Setup (Your Next Steps)

### 1. Create App Service
- [ ] Go to [portal.azure.com](https://portal.azure.com)
- [ ] Create new Web App resource
- [ ] Configure: Python 3.9, Linux, East US, Free F1 tier
- [ ] Use unique name: `fraudguard-app-[your-suffix]`

### 2. Configure Deployment
- [ ] Go to Deployment Center
- [ ] Connect to GitHub repository: `rds-on-rails/fraudguard-mvp`
- [ ] Set branch: `main`
- [ ] Enable continuous deployment

### 3. Application Settings
- [ ] Add `WEBSITES_PORT = 8000`
- [ ] Add `SCM_DO_BUILD_DURING_DEPLOYMENT = true`
- [ ] Add `ENABLE_ORYX_BUILD = true`
- [ ] Set startup command: `python startup.py`

### 4. Deployment Verification
- [ ] Wait for deployment to complete (5-10 minutes)
- [ ] Check deployment logs in Azure Portal
- [ ] Test app URL: `https://[your-app-name].azurewebsites.net`
- [ ] Run verification script: `python verify_deployment.py [your-url]`

### 5. Functional Testing
- [ ] App loads successfully
- [ ] Generate sample transaction data
- [ ] Run fraud detection analysis
- [ ] Verify visualizations display correctly
- [ ] Test CSV export functionality

## Troubleshooting Commands

If you encounter issues, use these commands locally:

```bash
# Test app locally first
streamlit run app.py

# Check if all dependencies install correctly
pip install -r requirements.txt

# Verify Python version compatibility
python --version
```

## Expected Results
- **App URL**: `https://[your-app-name].azurewebsites.net`
- **Load Time**: 2-5 seconds for initial page
- **Features**: Full fraud detection ML pipeline
- **Data**: Auto-generates 1000 sample transactions
- **Export**: CSV download functionality

## Support
- Check Azure Portal logs if deployment fails
- Verify GitHub repository connection
- Ensure all application settings are configured
- Monitor resource usage on Free F1 tier (60 min/day limit)
