# 🛡️ FraudGuard MVP - User Guide

## Overview
FraudGuard MVP is a real-time transaction anomaly detection system powered by machine learning. This comprehensive guide covers all features and functionality available in the application.

---

## 🚀 Getting Started

### Accessing the Application
- **Local Development**: `http://localhost:8503`
- **Streamlit Cloud**: `https://fraudguard-mvp.streamlit.app` (after deployment)

### System Requirements
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for cloud deployment
- No additional software installation required

---

## 📋 Main Interface Overview

The FraudGuard MVP interface consists of four main sections:

1. **Header Section**: Application branding and title
2. **Sidebar**: Model information and quick actions
3. **Main Content Area**: Data input and analysis tools
4. **Results Section**: Transaction data display and visualizations

---

## 🔧 Sidebar Features

### Model Information Panel
- **Model Status**: Shows whether the ML model is trained or not
- **Training Statistics**: 
  - Number of training samples used
  - Feature count in the model
  - Contamination rate (percentage of anomalies expected)
  - Model type (Isolation Forest)

### Quick Actions
- **🎯 Auto-Train Model**: Automatically generates 1000 sample transactions and trains the ML model
  - Click this button first when using the app
  - Training typically takes a few seconds
  - Model status will update to "✅ Model Trained" when complete

---

## 📊 Data Input Methods

### Method 1: Generate Sample Data
**Purpose**: Create synthetic transaction data for testing and demonstration

**Steps**:
1. Select "Generate Sample Data" radio button
2. Use the slider to choose number of samples (10-1000)
3. Click "🎲 Generate Data" button
4. System generates realistic transaction data with:
   - User IDs (UUID format)
   - Transaction amounts ($1.00 - $10,000.00)
   - Timestamps (last 30 days)
   - Locations (city names)
   - Device IDs (UUID format)

**Use Cases**:
- Testing the fraud detection system
- Demonstrating capabilities to stakeholders
- Training and learning how the system works

### Method 2: Upload CSV File
**Purpose**: Analyze real transaction data from your organization

**Requirements**:
Your CSV file must contain these exact column names:
- `user_id`: Unique identifier for each user
- `amount`: Transaction amount (numeric)
- `timestamp`: Date and time of transaction
- `location`: Transaction location
- `device_id`: Device identifier used for transaction

**Steps**:
1. Select "Upload CSV File" radio button
2. Click "Browse files" or drag-and-drop your CSV file
3. System validates the file structure
4. If successful, displays confirmation with record count
5. If columns are missing, shows error message with required fields

**File Format Example**:
```csv
user_id,amount,timestamp,location,device_id
user123,150.50,2024-01-15 10:30:00,New York,device456
user789,75.25,2024-01-15 11:45:00,Los Angeles,device123
```

---

## 🎯 Fraud Detection Process

### Prerequisites
1. **Model must be trained**: Use "🎯 Auto-Train Model" button
2. **Data must be loaded**: Either generate sample data or upload CSV

### Running Fraud Detection
1. Click "🔍 Run Fraud Detection" button (appears in right column)
2. System processes all transactions through the ML model
3. Each transaction receives:
   - **Fraud Flag**: 0 (Normal) or 1 (Suspicious)
   - **Prediction Score**: Numerical score indicating anomaly level
   - **Risk Level**: "High Risk" or "Low Risk" classification
   - **Status**: "Normal" or "Suspicious" text label

### Understanding Results
- **Normal Transactions**: Low prediction scores, typical patterns
- **Suspicious Transactions**: High prediction scores, unusual patterns
- **Prediction Score**: Higher values indicate greater likelihood of fraud
- **Risk Levels**: Based on score thresholds for easy interpretation

---

## 📈 Results Display & Analysis

### Summary Metrics Dashboard
Four key metrics displayed at the top:
1. **Total Transactions**: Complete count of processed records
2. **Suspicious**: Number and percentage of flagged transactions
3. **Normal**: Count of transactions marked as legitimate
4. **Average Fraud Score**: Mean prediction score for suspicious transactions

### Data Table Features
- **Color Coding**: 
  - 🔴 Red background: Suspicious transactions
  - 🟢 Green background: Normal transactions
- **Sortable Columns**: Click column headers to sort data
- **Scrollable View**: Handle large datasets efficiently
- **Real-time Updates**: Refreshes automatically after analysis

### Filtering Options

#### 1. Suspicious Transactions Filter
- **Checkbox**: "Show only suspicious transactions"
- **Purpose**: Focus on potentially fraudulent activities
- **Result**: Hides normal transactions from view

#### 2. Risk Level Filter
- **Dropdown**: "All", "High Risk", "Low Risk"
- **Purpose**: Further refine suspicious transaction analysis
- **Combination**: Works with suspicious-only filter

### Data Export
- **📥 Download Results as CSV**: Export filtered results
- **Filename Format**: `fraudguard_results_YYYYMMDD_HHMMSS.csv`
- **Content**: Includes all visible columns and applied filters
- **Use Cases**: 
  - Further analysis in Excel/other tools
  - Reporting to management
  - Record keeping and audit trails

---

## 📊 Data Visualizations

### 1. Transaction Status Distribution (Pie Chart)
- **Location**: Left column of visualization section
- **Purpose**: Shows proportion of normal vs. suspicious transactions
- **Colors**: Green for normal, red for suspicious
- **Interactive**: Hover for exact counts and percentages

### 2. Amount Distribution by Status (Box Plot)
- **Location**: Right column of visualization section
- **Purpose**: Compare transaction amounts between normal and suspicious
- **Insights**: 
  - Identifies if fraud correlates with transaction size
  - Shows outliers in each category
  - Reveals spending pattern differences

### 3. Prediction Score Distribution (Histogram)
- **Location**: Full width below other charts
- **Purpose**: Visualize the distribution of ML model confidence scores
- **Features**:
  - Color-coded by transaction status
  - Shows score ranges for normal vs. suspicious
  - Helps understand model decision boundaries

### Chart Interactions
- **Zoom**: Click and drag to zoom into specific areas
- **Pan**: Hold and drag to move around zoomed charts
- **Hover**: Display exact values and additional information
- **Legend**: Click to show/hide specific categories

---

## 🔍 Advanced Features

### Machine Learning Model Details
- **Algorithm**: Isolation Forest (unsupervised anomaly detection)
- **Training Data**: 1000 synthetic transactions with realistic patterns
- **Feature Engineering**: 
  - Time-based features (hour, day of week, weekend indicator)
  - Amount transformations (logarithmic scaling)
  - Location encoding
  - User behavior patterns
- **Contamination Rate**: 10% (expects 10% of transactions to be anomalies)

### Session Management
- **Data Persistence**: Your data stays loaded during the session
- **Model State**: Trained model remains available until page refresh
- **Results Caching**: Previous analysis results are preserved

### Performance Optimization
- **Efficient Processing**: Handles up to 1000 transactions smoothly
- **Real-time Analysis**: Results appear within seconds
- **Responsive Design**: Works on desktop, tablet, and mobile devices

---

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### "Model Not Trained" Message
- **Problem**: ML model hasn't been initialized
- **Solution**: Click "🎯 Auto-Train Model" in the sidebar
- **Wait Time**: 5-10 seconds for training completion

#### CSV Upload Errors
- **Problem**: "Missing columns" error message
- **Solution**: Ensure your CSV has exact column names: `user_id`, `amount`, `timestamp`, `location`, `device_id`
- **Check**: Column names are case-sensitive and must match exactly

#### No Data Displayed
- **Problem**: Empty results table
- **Solution**: 
  1. Verify data was loaded successfully
  2. Check if filters are too restrictive
  3. Try generating sample data first

#### Slow Performance
- **Problem**: App responds slowly
- **Solutions**:
  - Reduce number of transactions (use smaller datasets)
  - Refresh the browser page
  - Check internet connection for cloud deployment

### Browser Compatibility
- **Recommended**: Chrome, Firefox, Safari, Edge (latest versions)
- **JavaScript**: Must be enabled
- **Cookies**: Required for session management

---

## 📚 Use Cases & Applications

### 1. Financial Services
- **Credit Card Fraud**: Detect unusual spending patterns
- **Banking Transactions**: Monitor account activity
- **Payment Processing**: Flag suspicious payments
- **Risk Assessment**: Evaluate transaction risk levels

### 2. E-commerce Platforms
- **Purchase Monitoring**: Identify fraudulent orders
- **Account Security**: Detect compromised accounts
- **Payment Validation**: Verify transaction legitimacy
- **Chargeback Prevention**: Reduce fraudulent transactions

### 3. Business Intelligence
- **Pattern Analysis**: Understand normal vs. abnormal behavior
- **Risk Management**: Quantify fraud exposure
- **Compliance Reporting**: Generate audit trails
- **Performance Metrics**: Track fraud detection effectiveness

### 4. Research & Development
- **Algorithm Testing**: Evaluate fraud detection approaches
- **Data Science**: Prototype ML solutions
- **Academic Research**: Study anomaly detection methods
- **Proof of Concept**: Demonstrate fraud detection capabilities

---

## 🔒 Security & Privacy

### Data Handling
- **Local Processing**: All analysis happens in your browser session
- **No Data Storage**: Transaction data is not permanently stored
- **Session-based**: Data cleared when you close the browser
- **Privacy-First**: No personal information is transmitted or stored

### Synthetic Data
- **Generated Data**: Sample data is completely artificial
- **No Real Information**: Contains no actual personal or financial data
- **Safe Testing**: Use for demonstrations without privacy concerns
- **Realistic Patterns**: Mimics real transaction characteristics

---

## 📞 Support & Resources

### Getting Help
- **Documentation**: This user guide covers all features
- **Error Messages**: App provides specific guidance for issues
- **Status Indicators**: Clear visual feedback for all actions
- **Tooltips**: Hover over elements for additional help

### Best Practices
1. **Always train the model first** before analyzing data
2. **Start with sample data** to understand the system
3. **Use filters** to focus on relevant transactions
4. **Export results** for further analysis
5. **Check data quality** before uploading CSV files

### Performance Tips
- **Optimal Dataset Size**: 100-500 transactions for best performance
- **Regular Refresh**: Refresh browser if experiencing issues
- **Filter Usage**: Use filters to manage large datasets
- **Export Early**: Download results before applying multiple filters

---

## 🔄 Workflow Examples

### Example 1: Quick Fraud Check
1. Click "🎯 Auto-Train Model" (wait for completion)
2. Select "Generate Sample Data"
3. Set slider to 100 transactions
4. Click "🎲 Generate Data"
5. Click "🔍 Run Fraud Detection"
6. Review results in the data table
7. Use "Show only suspicious transactions" to focus on fraud
8. Download results if needed

### Example 2: Analyzing Your Data
1. Click "🎯 Auto-Train Model" (wait for completion)
2. Select "Upload CSV File"
3. Choose your transaction file
4. Verify successful upload
5. Click "🔍 Run Fraud Detection"
6. Review summary metrics
7. Apply filters as needed
8. Examine visualizations
9. Export filtered results

### Example 3: Comparative Analysis
1. Train model with sample data
2. Generate 200 sample transactions
3. Run fraud detection
4. Note the fraud percentage
5. Apply different filters
6. Compare results across different views
7. Use visualizations to understand patterns
8. Export different filtered views for comparison

---

## 📊 Understanding the Results

### Interpreting Prediction Scores
- **Range**: Typically -0.5 to 0.5
- **Higher Values**: More likely to be fraudulent
- **Lower Values**: More likely to be normal
- **Threshold**: Usually around 0.0 separates normal from suspicious

### Risk Level Classifications
- **High Risk**: Prediction score above threshold, requires immediate attention
- **Low Risk**: Prediction score below threshold, likely legitimate
- **Borderline Cases**: Scores near threshold may need manual review

### Statistical Insights
- **False Positives**: Normal transactions flagged as suspicious
- **False Negatives**: Fraudulent transactions marked as normal
- **Model Accuracy**: Depends on training data quality and feature engineering
- **Continuous Improvement**: Model performance improves with more training data

---

*This user guide covers all features available in FraudGuard MVP v1.0. For technical support or feature requests, please refer to the project documentation or contact the development team.*
