# 🚀 FraudGuard - Azure App Service Deployment Guide

## 📋 Pre-Deployment Status
✅ **Azure configuration files created and committed**  
✅ **Code pushed to GitHub repository**: `rds-on-rails/fraudguard-mvp`  
✅ **Streamlit app optimized for Azure App Service**  

## 🌐 Azure Portal Deployment Steps

### Step 1: Create Azure App Service

1. **Navigate to Azure Portal**: [portal.azure.com](https://portal.azure.com)
2. **Create Resource** → Search "Web App" → Click "Create"
3. **Configure Basic Settings**:
   ```
   Subscription: [Your Azure Subscription]
   Resource Group: fraudguard-rg (create new)
   Name: fraudguard-app-[unique-suffix]
   Publish: Code
   Runtime stack: Python 3.9
   Operating System: Linux
   Region: East US
   App Service Plan: Create new (Free F1)
   ```

### Step 2: Configure Deployment Source

1. **Go to your App Service** → "Deployment Center"
2. **Select GitHub**:
   - Repository: `rds-on-rails/fraudguard-mvp`
   - Branch: `main`
   - Build provider: App Service build service
3. **Save configuration**

### Step 3: Application Settings

Navigate to **Configuration** → **Application settings** and add:

| Name | Value | Description |
|------|-------|-------------|
| `WEBSITES_PORT` | `8000` | Port for Streamlit app |
| `SCM_DO_BUILD_DURING_DEPLOYMENT` | `true` | Enable build during deployment |
| `ENABLE_ORYX_BUILD` | `true` | Enable Oryx build system |
| `PYTHONPATH` | `/home/site/wwwroot` | Python path for modules |

### Step 4: Startup Configuration

1. **Configuration** → **General settings**
2. **Startup Command**: `python startup.py`
3. **Save changes**

## 🔧 Troubleshooting

### Common Issues & Solutions

**Issue**: App doesn't start
- **Check**: Application logs in "Log stream"
- **Solution**: Verify startup command and port configuration

**Issue**: Module import errors
- **Check**: requirements.txt includes all dependencies
- **Solution**: Restart the app service after configuration changes

**Issue**: Streamlit not accessible
- **Check**: WEBSITES_PORT is set to 8000
- **Solution**: Ensure startup.py uses correct port binding

## 📊 Expected App Features

Once deployed, your FraudGuard app will provide:

- **🛡️ Real-time fraud detection** using Isolation Forest ML
- **📈 Interactive dashboards** with Plotly visualizations
- **💾 Data processing** for transaction analysis
- **📤 CSV export** functionality
- **🎯 Anomaly scoring** and threshold configuration

## 🌐 Access Your Deployed App

**URL Format**: `https://[your-app-name].azurewebsites.net`

Example: `https://fraudguard-app-demo.azurewebsites.net`

## 📝 Post-Deployment Checklist

- [ ] Verify app loads successfully
- [ ] Test fraud detection functionality
- [ ] Check data generation and analysis
- [ ] Validate CSV export feature
- [ ] Monitor application logs
- [ ] Configure custom domain (optional)
- [ ] Set up monitoring and alerts

## 🔄 Updating Your App

To update your deployed app:
1. Make changes to your local code
2. Commit and push to GitHub
3. Azure will automatically redeploy from the main branch

## 💰 Cost Considerations

- **Free F1 Tier**: 60 minutes/day runtime limit
- **Basic B1**: $13.14/month for production use
- **Standard S1**: $56.94/month for higher performance

## 🆘 Support Resources

- [Azure App Service Documentation](https://docs.microsoft.com/en-us/azure/app-service/)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
- [Python on Azure App Service](https://docs.microsoft.com/en-us/azure/app-service/configure-language-python)
