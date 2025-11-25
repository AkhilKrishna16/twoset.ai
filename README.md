# TwoSet Backend API

A FastAPI-based backend service for stock price prediction using deep learning models. This service provides REST API endpoints for user authentication, API key management, and stock price forecasting using various neural network architectures.

## 🚀 Features

- **Multiple Deep Learning Models**: Support for LSTM, GRU, Bidirectional GRU, and RNN architectures
- **Stock Price Prediction**: Predict future stock prices for any ticker symbol
- **User Authentication**: Secure API key-based authentication system
- **Model Fine-tuning**: Automatic fine-tuning of base models for specific stock tickers
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Feature Engineering**: Includes technical indicators (RSI, SMA, Volatility, Returns)
- **Caching**: Efficient model caching for improved performance

## 🏗️ Architecture

The backend uses FastAPI with the following key components:

- **FastAPI**: Modern, fast web framework for building APIs
- **Keras/TensorFlow**: Deep learning models for stock price prediction
- **Supabase**: Database and authentication service
- **yfinance**: Yahoo Finance API for stock data
- **scikit-learn**: Data preprocessing and scaling

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Supabase account (for database and authentication)

## 🔧 Installation

1. **Clone the repository** (if applicable) or navigate to the backend directory:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn supabase python-dotenv keras tensorflow yfinance pandas scikit-learn numpy pydantic
   ```

   Or create a `requirements.txt` file with the following:
   ```
   fastapi>=0.104.0
   uvicorn>=0.24.0
   supabase>=2.0.0
   python-dotenv>=1.0.0
   keras>=2.14.0
   tensorflow>=2.14.0
   yfinance>=0.2.28
   pandas>=2.0.0
   scikit-learn>=1.3.0
   numpy>=1.24.0
   pydantic>=2.0.0
   ```

4. **Set up environment variables**:
   Create a `.env` file in the backend directory:
   ```env
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   TABLE=your_table_name
   ```

5. **Ensure model files are present**:
   Make sure the following model files exist in the `models/` directory:
   - `gru.keras`
   - `lstm.keras`
   - `bi_gru.keras`
   - `rnn.keras`

## 🔑 Environment Variables

Create a `.env` file with the following variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_KEY` | Your Supabase API key | Yes |
| `TABLE` | Name of the database table for users | Yes |

## 🚀 Running the Server

1. **Start the FastAPI server**:
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

   Or using Python:
   ```bash
   python -m uvicorn app:app --reload
   ```

2. **Access the API documentation**:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 📡 API Endpoints

### 1. Root Endpoint
- **GET** `/`
- **Description**: Health check endpoint
- **Response**:
  ```json
  {
    "message": "Initializing TwoSet!",
    "db_auth": "database_auth_url"
  }
  ```

### 2. User Signup
- **POST** `/signup_user`
- **Description**: Register a new user
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "api_key": ""
  }
  ```
- **Response**:
  ```json
  {
    "message": "User was signed up successfully!",
    "data": {...}
  }
  ```

### 3. Get API Key
- **POST** `/get_api_key`
- **Description**: Generate or retrieve an API key for an existing user
- **Request Body**:
  ```json
  {
    "email": "user@example.com",
    "api_key": ""
  }
  ```
- **Response**:
  ```json
  {
    "message": "API Key generated successfully!",
    "key": "uuid-api-key-here"
  }
  ```

### 4. Stock Price Prediction
- **GET** `/model/predict`
- **Description**: Predict future stock prices
- **Authentication**: Requires Bearer token in Authorization header
- **Query Parameters**:
  - `ticker` (string, required): Stock ticker symbol (e.g., "AAPL", "MSFT")
  - `model_type` (string, optional): Model type - "lstm", "gru", "bi_gru", or "rnn". Default: "lstm"
  - `batch_size` (int, optional): Sequence length for prediction. Default: 60
  - `prediction_length` (int, optional): Number of future days to predict. Default: 50

- **Headers**:
  ```
  Authorization: Bearer your_api_key_here
  ```

- **Response**:
  ```json
  {
    "predictions": [100.5, 101.2, 102.1, ...]
  }
  ```

## 💻 Usage Examples

### Python Example

```python
import requests

# 1. Sign up a user
signup_data = {
    "email": "user@example.com",
    "api_key": ""
}
response = requests.post("http://localhost:8000/signup_user", json=signup_data)
print(response.json())

# 2. Get API key
api_key_response = requests.post("http://localhost:8000/get_api_key", json=signup_data)
api_key = api_key_response.json()["key"]
print(f"API Key: {api_key}")

# 3. Make a prediction
headers = {"Authorization": f"Bearer {api_key}"}
params = {
    "ticker": "AAPL",
    "model_type": "lstm",
    "batch_size": 60,
    "prediction_length": 50
}
prediction_response = requests.get(
    "http://localhost:8000/model/predict",
    headers=headers,
    params=params
)
predictions = prediction_response.json()["predictions"]
print(f"Predicted prices: {predictions}")
```

### cURL Example

```bash
# Sign up
curl -X POST "http://localhost:8000/signup_user" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "api_key": ""}'

# Get API key
curl -X POST "http://localhost:8000/get_api_key" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "api_key": ""}'

# Predict (replace YOUR_API_KEY with actual key)
curl -X GET "http://localhost:8000/model/predict?ticker=AAPL&model_type=lstm&prediction_length=50" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## 🤖 Machine Learning Models

### Supported Model Types

1. **LSTM (Long Short-Term Memory)**
   - Best for capturing long-term dependencies
   - Default model type

2. **GRU (Gated Recurrent Unit)**
   - Faster training than LSTM
   - Good performance with less complexity

3. **Bidirectional GRU**
   - Processes sequences in both directions
   - Can capture more context

4. **RNN (Simple Recurrent Network)**
   - Basic recurrent architecture
   - Simpler but less powerful

### Model Features

The models use the following features for prediction:
- **Close Price**: Closing stock price
- **Volume**: Trading volume
- **Returns**: Percentage change in price
- **SMA_20**: 20-day Simple Moving Average
- **SMA_50**: 50-day Simple Moving Average
- **RSI**: Relative Strength Index (14-day period)
- **Volatility**: Rolling standard deviation of returns (20-day window)

### Model Fine-tuning

When a prediction is requested for a ticker:
1. The system checks if a fine-tuned model exists for that ticker
2. If not, it fine-tunes the base model on the ticker's historical data
3. The fine-tuned model is cached for future requests
4. Fine-tuning uses transfer learning: base layers are frozen, only the output layer is trained

## 📁 Project Structure

```
backend/
├── app.py                  # Main FastAPI application
├── test.py                 # Test utilities
├── models/                 # Pre-trained model files
│   ├── gru.keras
│   ├── lstm.keras
│   ├── bi_gru.keras
│   └── rnn.keras
├── model_files/            # Model training scripts
│   ├── generate_v1.py      # Model generation script
│   ├── gru.ipynb
│   ├── lstm.ipynb
│   ├── bi_gru.ipynb
│   └── rnn.ipynb
└── README.md              # This file
```

## 🔒 Security

- API key-based authentication for prediction endpoints
- Bearer token authentication
- Environment variables for sensitive credentials
- Input validation using Pydantic models

## ⚠️ Limitations

- Predictions are based on historical data and technical indicators
- Stock market predictions are inherently uncertain
- Models are fine-tuned on 5 years of historical data
- Fine-tuning uses a single epoch for speed

## 🐛 Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure all `.keras` model files are present in the `models/` directory
   - Check file paths are correct

2. **Supabase connection errors**
   - Verify `SUPABASE_URL` and `SUPABASE_KEY` in `.env` file
   - Ensure the table name matches your Supabase table

3. **Import errors**
   - Activate your virtual environment
   - Reinstall dependencies: `pip install -r requirements.txt`

4. **Ticker data not found**
   - Verify the ticker symbol is valid
   - Check internet connection for yfinance data fetching

## 📝 Notes

- The initial startup may take time as models are loaded and fine-tuned for common tickers (AAPL, MSFT)
- Predictions are scaled using MinMaxScaler and then inverse transformed
- The system uses a rolling window approach for multi-step predictions

## 🔄 Future Enhancements

- Add more technical indicators
- Implement model ensemble predictions
- Add prediction confidence intervals
- Support for different time intervals (hourly, weekly)
- Historical prediction accuracy tracking
- Rate limiting for API endpoints

## 📄 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 📧 Contact

[Add contact information here]

