# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import pandas as pd
from sqlalchemy import create_engine
from pydantic import BaseModel


import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from create_database import execute

app = FastAPI()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

origins = [
    "http://localhost",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Połączenie z bazą danych SQLite
logging.info("Pomyślnie połączono z bazą danych SQLite...")
engine = create_engine('sqlite:///example.db')

@app.get("/")
async def main():
    df = pd.read_sql_query("SELECT business  FROM my_table", engine)
    print(df['business'])
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
        return {"error": "Wystąpił błąd podczas przetwarzania danych. linijka 60 "}

@app.get("/getArticlesAsBusiness")
async def articles_as_business():
    try:
        engine = create_engine('sqlite:///example.db')
        logging.info("Pomyślnie połączono z bazą danych SQLite...")

        # Odczytanie danych z bazy SQLite
        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT business, emails, url, keywords FROM my_table", engine)
        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        return df.to_dict('records')

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd podczas przetwarzania danych."}

@app.get("/getArticlesAsInvestors")
async def articles_as_investors():
    try:
        engine = create_engine('sqlite:///example.db')
        logging.info("Pomyślnie połączono z bazą danych SQLite...")

        # Odczytanie danych z bazy SQLite
        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT investors, emails, url, keywords FROM my_table", engine)
        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        return df.to_dict('records')

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd podczas przetwarzania danych."}
class Query(BaseModel):
    query: str


@app.post("/getArticlesAsBusiness")
async def compareText(query: Query):
    try:

        engine = create_engine('sqlite:///example.db')
        logging.info("Superowo połączono z bazą danych SQLite...")

        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT business, emails, url, keywords, abstract_embedding FROM my_table", engine)
        df['abstract_embedding'] = df['abstract_embedding'].apply(
            lambda x: np.frombuffer(x, dtype=np.float32).reshape(-1)
        )

        query_embedding = extract_features(query.query).detach().cpu().numpy()
        # print(query_embedding.shape)
        # print(df['abstract_embedding'][0].shape)

        similarities = [(item, get_similarities(query_embedding, item)) for _, item in df.iterrows()]
        # print(similarities)
        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame([sorted_result[0] for sorted_result in sorted_results])
        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        df = df[['business', 'emails', 'url', 'keywords']]
        return df.to_dict('records')
        # return None

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd w czasie przetwarzania danych."}


@app.post("/getArticlesAsInvestors")
async def compareText2(query: Query):
    try:

        engine = create_engine('sqlite:///example.db')
        logging.info("Superowo połączono z bazą danych SQLite...")

        logging.info("Odczytywanie danych z bazy SQLite...")
        df = pd.read_sql_query("SELECT investors, emails, url, keywords, abstract_embedding FROM my_table", engine)
        df['abstract_embedding'] = df['abstract_embedding'].apply(
            lambda x: np.frombuffer(x, dtype=np.float32).reshape(-1)
        )

        query_embedding = extract_features(query.query).detach().cpu().numpy()
        # print(query_embedding.shape)
        # print(df['abstract_embedding'][0].shape)

        similarities = [(item, get_similarities(query_embedding, item)) for _, item in df.iterrows()]
        # print(similarities)
        sorted_results = sorted(similarities, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame([sorted_result[0] for sorted_result in sorted_results])
        # Zwrócenie danych w formacie JSON
        logging.info("Zwracanie danych w formacie JSON...")
        df = df[['investors', 'emails', 'url', 'keywords']]
        return df.to_dict('records')
        # return None

    except Exception as e:
        logging.error(f"Wystąpił błąd: {e}")
        return {"error": "Wystąpił błąd w czasie przetwarzania danych."}

model_sim = SentenceTransformer("all-MiniLM-L6-v2")

def get_similarities(e1, e2):
    # print(e2)
    # logging.info(e2.shape)
    # e1 = np.mean(e1, axis=1)
    # print(e1.shape)
    # print()
    # print(e2.shape)
    # if e1.ndim == 1:
    e1 = e1.reshape(1, -1)
    # if e2.ndim == 1:
    e2 = e2['abstract_embedding'].reshape(1, -1)
    similarities = model_sim.similarity(e2, e1)
    return similarities


def extract_features(text):
    model_name = "allenai/scibert_scivocab_uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Dodaj padding i truncation
    inputs = tokenizer(text,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=512)  # Standardowa długość dla BERT

    outputs = model(**inputs)

    # Weź średnią z ostatniej warstwy ukrytej
    # Wymiar: (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
    embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings[0]  # Zwróć wektor dla pierwszego (i jedynego) elementu batcha
# def extract_features(text):
#     model_name = "allenai/scibert_scivocab_uncased"
#
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModel.from_pretrained(model_name)
#
#     input_ids = tokenizer.encode(text, return_tensors="pt")
#     output = model(input_ids)[0]
#     return output

@app.get("/create_database")
async def create_database():
    execute()