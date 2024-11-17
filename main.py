# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import pandas as pd
from sqlalchemy import create_engine
from create_database import execute

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Połączenie z bazą danych SQLite
logging.info("Pomyślnie połączono z bazą danych SQLite...")
engine = create_engine('sqlite:///example.db')

@app.get("/")
async def main():
    return {"message": "Hello World"}

@app.get("/data")
async def get_data():
    try:
        engine = create_engine('sqlite:///example.db')
        logging.info("Pomyślnie połączono z bazą danych SQLite...")

        # Odczytanie danych z bazy SQLite
        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT title FROM my_table", engine)

        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        return df.to_dict('records')

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd podczas przetwarzania danych."}

@app.get("/getArticlesAsBusiness")
async def articles_as_business():
    try:
        engine = create_engine('sqlite:///example.db')
        logging.info("Pomyślnie połączono z bazą danych SQLite...")

        # Odczytanie danych z bazy SQLite
        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT business, authors, url FROM my_table", engine)
        df['keywords'] = df['authors']
        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        return df.to_dict('records')

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd podczas przetwarzania danych."}


@app.get("/create_database")
async def create_database():
    execute()
