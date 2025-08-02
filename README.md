# 🛡️ FraudGuard MVP

A real-time fraud detection system built with FastAPI, Streamlit, and scikit-learn. This MVP uses an Isolation Forest machine learning model to detect anomalous transaction patterns that may indicate fraudulent activity.

## 🚀 Features

### Backend (FastAPI)
- **REST API** with `/predict` endpoint for batch fraud detection
- **Isolation Forest** model from scikit-learn for anomaly detection
- **Automatic model training** on startup with synthetic data
- **Health checks** and model retraining endpoints
- **CORS support** for frontend integration

### Frontend (Streamlit)
- **Interactive web interface** for fraud detection analysis
- **CSV file upload** for custom transaction data
- **Random data generation** using Faker library
- **Real-time fraud detection** with visual results
- **Filtering and summary statistics** for analysis
- **Data visualization** with Plotly charts
- **Export results** to CSV

### ML Model
- **Isolation Forest** algorithm for unsupervised anomaly detection
- **Feature engineering** including time-based and user behavior features
- **Synthetic training data** generation for model initialization
- **Configurable contamination rate** for fraud detection sensitivity

## 📊 Transaction Data Schema

The system processes transactions with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Unique user identifier |
| `amount` | float | Transaction amount |
| `timestamp` | datetime | Transaction timestamp |
| `location` | string | Transaction location |
| `device_id` | string | Device identifier |

## 🏗️ Project Structure

```
FraudGuard/
├── backend/                 # FastAPI application
│   ├── __init__.py
│   └── main.py             # API endpoints and server
├── frontend/               # Streamlit application
│   ├── __init__.py
│   └── app.py              # Web interface
├── models/                 # ML models and logic
│   ├── __init__.py
│   └── fraud_detector.py   # Isolation Forest implementation
├── utils/                  # Utilities and helpers
│   ├── __init__.py
│   └── data_simulator.py   # Data generation and feature engineering
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Multi-service orchestration
└── README.md              # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.11+ 
- Docker (optional, for containerized deployment)
- Git

### Option 1: Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FraudGuard
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend API**
   ```bash
   cd backend
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`
   
   API documentation: `http://localhost:8000/docs`

5. **Start the frontend (in a new terminal)**
   ```bash
   cd frontend
   streamlit run app.py
   ```
   
   The web interface will be available at `http://localhost:8501`

### Option 2: Docker Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FraudGuard
   ```

2. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

   This will start both services:
   - Backend API: `http://localhost:8000`
   - Frontend UI: `http://localhost:8501`

3. **Stop the services**
   ```bash
   docker-compose down
   ```

### Option 3: Single Docker Container

1. **Build the Docker image**
   ```bash
   docker build -t fraudguard .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 -p 8501:8501 fraudguard
   ```

## 🎯 Usage Guide

### 1. Starting the Application

After following the setup instructions, you'll have:
- **Backend API** running on port 8000
- **Frontend UI** running on port 8501

### 2. Using the Web Interface

1. **Open your browser** and navigate to `http://localhost:8501`

2. **Choose your data source:**
   - **Upload CSV**: Upload a CSV file with transaction data
   - **Generate Random Data**: Create synthetic transactions for testing

3. **Run fraud detection:**
   - Click "Run Fraud Detection" to analyze the transactions
   - View results in real-time with summary statistics

4. **Analyze results:**
   - Filter transactions by fraud status
   - View detailed transaction tables
   - Explore interactive visualizations
   - Download results as CSV

### 3. Using the API Directly

#### Predict Fraud
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transactions": [
         {
           "user_id": "user_001",
           "amount": 150.50,
           "timestamp": "2024-01-15T10:30:00",
           "location": "New York",
           "device_id": "device_123"
         }
       ]
     }'
```

#### Check API Health
```bash
curl http://localhost:8000/health
```

#### Get Model Information
```bash
curl http://localhost:8000/model/info
```

#### Retrain Model
```bash
curl -X POST http://localhost:8000/retrain
```

## 📈 Understanding the Results

### Fraud Flags
- **0**: Normal transaction
- **1**: Suspicious/fraudulent transaction

### Key Metrics
- **Total Transactions**: Number of transactions analyzed
- **Suspicious Transactions**: Number flagged as potential fraud
- **Fraud Rate**: Percentage of transactions flagged as suspicious

### Visualizations
- **Distribution Chart**: Shows the ratio of normal vs. fraudulent transactions
- **Amount Distribution**: Box plot comparing transaction amounts by fraud status
- **Time Series**: Scatter plot showing transactions over time with fraud indicators

## 🔧 Configuration

### Model Parameters

You can adjust the fraud detection sensitivity by modifying the `contamination` parameter in `models/fraud_detector.py`:

```python
# Lower values = more sensitive (more false positives)
# Higher values = less sensitive (more false negatives)
contamination = 0.1  # Expects 10% of transactions to be anomalies
```

### Feature Engineering

The system automatically generates features from transaction data:

- **Time features**: Hour, day of week, weekend indicator
- **Amount features**: Original amount and log-transformed amount
- **User behavior**: Average amount, transaction count, amount variability
- **Device patterns**: Transaction count per device

## 🧪 Testing

### Generate Test Data
```python
from utils.data_simulator import generate_transaction_data

# Generate 1000 test transactions
df = generate_transaction_data(1000)
print(df.head())
```

### Manual API Testing

Use the interactive API documentation at `http://localhost:8000/docs` to test endpoints manually.

## 🚀 Production Considerations

### Security
- [ ] Add API authentication and rate limiting
- [ ] Use HTTPS in production
- [ ] Implement proper CORS policies
- [ ] Add input validation and sanitization

### Performance
- [ ] Implement model caching and persistence
- [ ] Add database for transaction storage
- [ ] Optimize feature engineering pipeline
- [ ] Add monitoring and logging

### Scalability
- [ ] Use container orchestration (Kubernetes)
- [ ] Implement load balancing
- [ ] Add message queues for batch processing
- [ ] Consider model serving platforms

## 🛠️ Development

### Adding New Features

1. **New API endpoints**: Add to `backend/main.py`
2. **Frontend components**: Modify `frontend/app.py`
3. **ML models**: Extend `models/fraud_detector.py`
4. **Data utilities**: Add to `utils/data_simulator.py`

### Testing Changes

```bash
# Run backend tests
cd backend
python -m pytest

# Test frontend manually
cd frontend
streamlit run app.py
```

## 📋 Dependencies

### Core Libraries
- **FastAPI**: Modern web framework for APIs
- **Streamlit**: Web app framework for data science
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing

### Additional Libraries
- **Faker**: Generate fake data for testing
- **Plotly**: Interactive visualizations
- **requests**: HTTP library for API calls
- **uvicorn**: ASGI server for FastAPI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Backend not starting:**
- Check if port 8000 is available
- Verify Python dependencies are installed
- Check logs for import errors

**Frontend can't connect to backend:**
- Ensure backend is running on port 8000
- Check firewall settings
- Verify API URL in frontend configuration

**Model training errors:**
- Check pandas and scikit-learn versions
- Verify sufficient memory for model training
- Check data format and features

**Docker issues:**
- Ensure Docker is running
- Check port conflicts (8000, 8501)
- Verify Docker Compose version compatibility

### Getting Help

1. Check the [API documentation](http://localhost:8000/docs)
2. Review application logs
3. Open an issue on GitHub
4. Check the troubleshooting section above

---

**FraudGuard MVP** - Built with ❤️ using FastAPI, Streamlit, and scikit-learn
