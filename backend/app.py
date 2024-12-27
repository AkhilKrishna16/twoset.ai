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
import asyncio 
import yfinance as yf

class User(BaseModel):
    email: EmailStr
    api_key: str
    
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
    models['base_rnn'] = keras.models.load_model('./models/rnn.keras')
    models['base_bi_gru'] = keras.models.load_model('./models/bidirectional-gru.keras')
    common_tickers = ['AAPL', 'MSFT']
    for ticker in common_tickers:
        for model in ['gru', 'lstm', 'rnn', 'bi_gru']:
            val = fine_tune_model(models[f'base_{model}'], ticker)
            models[f"{ticker}_{model}"] = val
            print(models[f"{ticker}_{model}"])
    yield 

app = FastAPI(lifespan=lifespan)

def get_ticker_data(ticker: str):
    stock_data = yf.Ticker(ticker).history(period='1y')
    X_train = stock_data[['Open', 'Close', 'Low', 'High']].values[:-1]
    y_train = stock_data['Close'].values[1:]
    data = {'X_train': X_train, 'y_train': y_train}
    return data

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
    
@lru_cache(maxsize=10)
def fine_tune_model(base_model, ticker: str):
    """
    

    Args:
        base_model (_type_): _description_
        stock_data (_type_): _description_
    """
    
    stock_data = get_ticker_data(ticker)
    model_copy = keras.models.clone_model(base_model)
    model_copy.set_weights(base_model.get_weights())
    
    for layer in model_copy.layers[:-1]: # everything but the last layer
        layer.trainable = False
        
    model_copy.compile(optimizer='adam', loss='mse')
    history = model_copy.fit(stock_data['X_train'], stock_data['y_train'], epochs=1, verbose=0)
    
    final_loss = history.history['loss'][-1]
    return model_copy, final_loss

    
    
