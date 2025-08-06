# 🛡️ FraudGuard MVP - Quick Reference Card

## 🚀 Getting Started (3 Steps)
1. **Train Model**: Click "🎯 Auto-Train Model" in sidebar
2. **Load Data**: Generate sample data OR upload CSV
3. **Detect Fraud**: Click "🔍 Run Fraud Detection"

---

## 📊 Key Features at a Glance

### Data Input Options
- **🎲 Generate Sample Data**: 10-1000 synthetic transactions
- **📁 Upload CSV**: Your own transaction data
- **Required CSV Columns**: `user_id`, `amount`, `timestamp`, `location`, `device_id`

### Analysis Tools
- **🔍 Fraud Detection**: ML-powered anomaly detection
- **📈 Summary Metrics**: Total, suspicious, normal, avg fraud score
- **🎯 Filtering**: Show suspicious only, filter by risk level
- **📥 Export**: Download results as CSV

### Visualizations
- **🥧 Pie Chart**: Transaction status distribution
- **📊 Box Plot**: Amount distribution by status
- **📈 Histogram**: Prediction score distribution

---

## 🎯 Quick Actions

| Action | Location | Purpose |
|--------|----------|---------|
| Auto-Train Model | Sidebar | Initialize ML model |
| Generate Data | Main area | Create test transactions |
| Upload CSV | Main area | Load your data |
| Run Detection | Right column | Analyze for fraud |
| Show Suspicious | Results section | Filter suspicious only |
| Download CSV | Results section | Export results |

---

## 🚨 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Model Not Trained" | Click "🎯 Auto-Train Model" |
| "Missing columns" | Check CSV has required columns |
| No results shown | Verify data loaded and model trained |
| Slow performance | Use smaller datasets (<500 records) |

---

## 📋 Workflow Checklist

### First Time Setup
- [ ] Click "🎯 Auto-Train Model"
- [ ] Wait for "✅ Model Trained" status
- [ ] Choose data input method
- [ ] Load or generate transaction data

### Running Analysis
- [ ] Verify model is trained
- [ ] Confirm data is loaded
- [ ] Click "🔍 Run Fraud Detection"
- [ ] Review summary metrics
- [ ] Apply filters if needed
- [ ] Export results if required

### Understanding Results
- [ ] Check fraud percentage in metrics
- [ ] Look for red-highlighted suspicious transactions
- [ ] Use visualizations to understand patterns
- [ ] Filter by risk level for focused analysis

---

## 💡 Pro Tips

- **Start Small**: Use 50-100 transactions for initial testing
- **Color Coding**: Red = Suspicious, Green = Normal
- **Export Early**: Download results before applying multiple filters
- **Check Patterns**: Use visualizations to understand fraud characteristics
- **Regular Training**: Retrain model periodically with new data patterns

---

*For detailed instructions, see the complete USER_GUIDE.md*
