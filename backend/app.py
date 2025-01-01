from fastapi import FastAPI, logger
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager
from supabase import create_client
from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr
import uuid
from functools import lru_cache
import keras
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class User(BaseModel):
    """
    User model for login and API key generation. 

    Args:
        BaseModel (class): extending BaseModel from pydantic for entries
    """
    email: EmailStr
    api_key: str
    
class MLInput(BaseModel):
    """
    Model for input into Machine Learning model. 

    Args:
        BaseModel (class): extending BaseModel from pydantic for entries
    """
    ticker: str
    model_type: str = 'lstm'
    batch_size: int = 60
    prediction_length: int = 50
    
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE")

global db
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize database with given URL and key. Initialize models with fine tuning. 

    Args:
        app (FastAPI): the app instance used by FastAPI
    """
    global db
    global models
    db = create_client(SUPABASE_URL, SUPABASE_KEY)
    models['base_gru'] = keras.models.load_model('./models/gru.keras')
    models['base_lstm'] = keras.models.load_model('./models/lstm.keras')
    models['base_bi_gru'] = keras.models.load_model('./models/bi_gru.keras')
    common_tickers = ['AAPL', 'MSFT']
    for ticker in common_tickers:
        for model in ['gru', 'lstm', 'bi_gru']:
            val = fine_tune_model(models[f'base_{model}'], ticker)
            models[f"{ticker}_{model}"] = val
    yield 

app = FastAPI(lifespan=lifespan)

def preprocess_ticker_data(ticker: str, timesteps=60):
    """
    Gather and prepares `yfinance` retrieved data based on the inputted ticker and timesteps. 

    Args:
        ticker (str): the ticker for which stock data which must be retrieved
        timesteps (int, optional): the batch_size at which the training set is gathered and trained upon. Defaults to 60.

    Returns:
        _type_: _description_
    """
    stock_data = yf.Ticker(ticker).history(period='5y', interval='1d')
    if stock_data.empty:
        return None
    training_set = stock_data['Open'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = scaler.fit_transform(training_set)
    X_train, y_train = [], []
    for i in range(timesteps, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-timesteps:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    return {
        'X_train': X_train,
        'y_train': y_train,
        'scaler': scaler,
    }

def postprocess_ticker_data(predictions, scaler):
    """
    Inverse-scales the prediction data for actual results.

    Args:
        predictions (list): the predictions that are given by the actual ticker data
        scaler (any): the scaler built on original predictions following (0,1)

    Returns:
        any: normalized predictions
    """
    return scaler.inverse_transform(predictions)

@app.get('/')
async def root() -> dict:
    """
    Initialize server.

    Returns:
        dict: Initialization message.
    """
    return {"message": "Initializing TwoSet!", "db_auth": db.auth_url}

@app.post('/signup_user')
async def add_user_to_db(user: User):
    """
    Updates table with new user if can be added.

    Args:
        user (User): the user that needs to be added (formatted with User structure)

    Raises:
        HTTPException: User already exists based on email
        HTTPException: Error with general adding of the user

    Returns:
        dict: json response of the data and indication of user being added
    """
    try:
        exists = db.table(TABLE_NAME).select('*').eq("email", user.email).execute()
        if exists.data:
            raise HTTPException(status_code=500, detail="User already exists")
        response = db.table(TABLE_NAME).insert(user.model_dump()).execute()
        if not response:
            raise HTTPException(status_code=400, detail="Error inserting data")
        return {"message": "User was signed up successfully!", "data": response.data}
    except Exception as e:
        return {"message": str(e)}

@app.post("/get_api_key")
async def get_api_key(user: User):
    """
    Generates API key for the user provided, if it exists.

    Args:
        user (User): the user that needs the API key

    Raises:
        HTTPException: User was not found.
        HTTPException: Error with general updating of user data for the api key 

    Returns:
        dict: whether the update was successful or not
    """
    try:
        existing_data = db.table(TABLE_NAME).select("*").eq("email", user.email).execute().data
        if not existing_data:
            raise HTTPException(status_code=500, detail="User not found")
        api_key = str(uuid.uuid4())
        response = db.table(TABLE_NAME).update({"api_key": api_key}).eq("email", user.email).execute()
        if not response:
            raise HTTPException(status_code=400, detail="Error updating data")
        return {"message": "API Key generated successfully!", "key": api_key}
    except Exception as e:
        return {"message": str(e)}
    
@lru_cache(maxsize=20)
def fine_tune_model(base_model, ticker: str):
    """
    Freezes layers up until last one and trains model on new ticker unless model_copy already present.

    Args:
        base_model (any): the loaded model based on the  
        ticker (str): _description_

    Returns:
        any: a model copy based on the trained model with the dataset
    """
    stock_data = preprocess_ticker_data(ticker)
    if stock_data:
        model_copy = keras.models.clone_model(base_model)
        model_copy.set_weights(base_model.get_weights())
        for layer in model_copy.layers[:-1]: 
            layer.trainable = False
        if len(model_copy.layers[-1].trainable_weights) == 0:
            model_copy.pop()
            model_copy.add(keras.layers.Dense(units=1, activation='linear'))
        model_copy.compile(optimizer='adam', loss='mse')
        postprocess_ticker_data(stock_data['y_train'].reshape(-1, 1), scaler=stock_data['scaler'])
        history = model_copy.fit(stock_data['X_train'], stock_data['y_train'], epochs=1, verbose=0)
        final_loss = history.history['loss'][-1]
        return model_copy, final_loss
    return None

@app.get('/model/predict')
def predict(inputs: MLInput):
    """
    Returns predictions based on the model, ticker data, and batch_size. 

    Args:
        inputs (MLInput): the MLInput model that contains fields with different training specifications

    Raises:
        HTTPException: non-existent model
        HTTPException: non-existent ticker data (invalid ticker -- i.e. ticker removed or non-existent data)
        HTTPException: training error (might be server error)

    Returns:
        json: the list of predictions in timesteps of size `prediction_size`.
    """
    try:
        model_key = f"{inputs.ticker}_{inputs.model_type}"
        if model_key in models:
            model, _ = models[model_key]
        else:
            updated_model = fine_tune_model(models[f'base_{inputs.model_type}'], inputs.ticker)
            if updated_model:
                models[model_key] = updated_model
                model, _ = updated_model
            else:
                raise HTTPException(status_code=404, detail='Model could not be found.')
        post_ticker_data = preprocess_ticker_data(inputs.ticker, inputs.batch_size)
        if not post_ticker_data:
            raise HTTPException(status_code=404, detail='Insufficient data for testing.')
        X_test = post_ticker_data['X_train'][-1:]
        scaler = post_ticker_data['scaler']
        predictions = []
        current_input = X_test
        for _ in range(inputs.prediction_length):
            pred = model.predict(current_input, verbose=0)
            predictions.append(pred[0, 0])
            current_input = np.append(current_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)
        predictions = np.array(predictions).reshape(-1, 1)
        predictions_processed = postprocess_ticker_data(predictions, scaler)
        return {'predictions': predictions_processed.flatten().tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))