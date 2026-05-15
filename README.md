# TwoSet Backend API

A FastAPI-based backend service for stock price prediction using deep learning models. This service provides REST API endpoints for user authentication, API key management, and stock price forecasting using various neural network architectures.

## 🚀 Features

- **Multiple Deep Learning Models**: Support for LSTM, GRU, Bidirectional GRU, and RNN architectures
- **Stock Price Prediction**: Predict future stock prices for any ticker symbol
- **DQN Trading Agent**: A from-scratch Double-DQN reinforcement-learning agent (PyTorch) that learns a discrete hold/buy/sell policy. Trains on a single ticker or a batch (e.g. the Mag-7) with one shared policy and reports per-ticker plus aggregate profit
- **User Authentication**: Secure API key-based authentication system
- **Model Fine-tuning**: Automatic fine-tuning of base models for specific stock tickers
- **Real-time Data**: Fetches live stock data using Yahoo Finance API
- **Feature Engineering**: Includes technical indicators (RSI, SMA, Volatility, Returns)
- **Caching**: Efficient model caching for improved performance

## 🏗️ Architecture

The backend uses FastAPI with the following key components:

- **FastAPI**: Modern, fast web framework for building APIs
- **Keras/TensorFlow**: Deep learning models for stock price prediction
- **PyTorch**: Reinforcement-learning agent (DQN trader) and supporting tensor ops
- **Supabase**: Database and authentication service
- **yfinance**: Yahoo Finance API for stock data
- **scikit-learn**: Data preprocessing and scaling
- **matplotlib**: Optional plotting for DQN evaluation runs

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
   pip install -r requirements.txt
   ```

   The pinned dependency list lives in `requirements.txt` at the repo root and
   currently includes:

   ```
   fastapi
   uvicorn[standard]
   python-dotenv
   pydantic[email]
   supabase
   scikit-learn
   keras
   tensorflow
   torch
   numpy
   yfinance
   matplotlib
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

## 🎯 DQN Trading Agent

The DQN trader is a standalone PyTorch script that trains a Double-DQN on
historical OHLC data fetched via yfinance and evaluates it greedily on a
held-out test split. It is not part of the FastAPI surface today; it runs
directly from the command line.

### Running

From the repository root:

```bash
# Single ticker
python -m backend.model_files.dqn_trader --ticker AAPL --episodes 25 --period 10y

# Batch across the Mag-7 with one shared policy
python -m backend.model_files.dqn_trader --tickers mag7 --episodes 25

# Arbitrary basket
python -m backend.model_files.dqn_trader --tickers AAPL,MSFT,NVDA
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker` | `AAPL` | Single ticker symbol passed to yfinance. Ignored if `--tickers` is set. |
| `--tickers` | `None` | Comma-separated tickers or a group alias. Currently `mag7` expands to `AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA`. |
| `--episodes` | `25` | Number of training episodes. In multi-ticker mode, each episode rolls one rollout per ticker (shuffled order). |
| `--period` | `10y` | yfinance history period (e.g. `5y`, `10y`, `max`). |
| `--plot` | off | If set, plot price + DQN actions and the portfolio-value curve on each test split. |
| `--plot-path` | `None` | If set together with `--plot`, save figures to this path. With multiple tickers the ticker symbol is appended to the filename. |

### Environment and reward

- Single-asset trading environment with a `WINDOW = 30` day rolling
  observation: the most recent log returns plus two portfolio-state scalars
  (cash ratio and position ratio). The observation is ticker-agnostic, which
  is what lets one shared policy train across a batch.
- Starting bankroll: `$10,000` per ticker. Commission: `1 bp` per trade.
- Reward: log return of mark-to-market portfolio value step-over-step, so
  cumulative reward equals total log return over an episode.
- Train/test split: the first 80% of each ticker's price history is used for
  training, the final 20% for greedy evaluation. The run reports per-ticker
  DQN test profit, the buy-and-hold baseline over the same window, the
  `$2,000` profit target check, and an aggregate (sum + average) across the
  batch.

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

5. **DQN Trading Agent (PyTorch)**
   - Double-DQN with a GRU price-window encoder and an MLP Q-head
   - Polyak (soft) target network updates and Huber (smooth L1) loss
   - Epsilon-greedy exploration with multiplicative decay
   - Replay buffer backed by pre-allocated NumPy ring buffers
   - Discrete action space: `0 = HOLD`, `1 = BUY`, `2 = SELL`
   - Reward is the log return of mark-to-market portfolio value per step

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
.
├── README.md                       # This file
├── LICENSE
├── requirements.txt                # Python dependencies (used by `pip install -r`)
└── backend/
    ├── app.py                      # Main FastAPI application
    ├── test.py                     # Test utilities
    ├── models/                     # Pre-trained Keras model files
    │   ├── gru.keras
    │   ├── lstm.keras
    │   ├── bi_gru.keras
    │   └── rnn.keras
    └── model_files/                # Model training notebooks + standalone scripts
        ├── gru.ipynb
        ├── lstm.ipynb
        ├── bidirectional-gru.ipynb
        ├── rnn.ipynb
        ├── dqn_trader.py           # PyTorch Double-DQN trading agent (CLI)
        ├── AAPL_stock_plot_lstm.png
        └── tsla_run.png            # Sample DQN evaluation plot
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
- The DQN trader is currently CLI-only and is not exposed via the FastAPI service

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

