from fastapi import FastAPI
from fastapi.exceptions import HTTPException
from contextlib import asynccontextmanager
from supabase import create_client
from dotenv import load_dotenv
import os
from pydantic import BaseModel, EmailStr
import uuid

class User(BaseModel):
    email: EmailStr
    api_key: str
    
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = os.getenv("TABLE")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize database with given URL and key.  

    Args:
        app (FastAPI): the app instance used by FastAPI
    """
    global db
    db = create_client(SUPABASE_URL, SUPABASE_KEY)
    yield 

app = FastAPI(lifespan=lifespan)

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