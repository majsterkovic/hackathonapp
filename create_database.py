import arxivscraper
import anthropic
import pandas as pd
import os
from anthropic.types import TextBlock
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import PyPDF2
from io import BytesIO
import sqlite3
from sqlalchemy import create_engine, types


def _extract_features(text, tokenizer, model):
    segment_length = 512
    segments = []
    for i in range(0, len(text), segment_length):
        segment = text[i:i + segment_length]
        if len(segment.split()) <= 512:
            segments.append(segment)
        else:
            # Podziel dłuższy segment na mniejsze
            subsegments = [" ".join(segment.split()[j:j + 512]) for j in range(0, len(segment.split()), 512)]
            segments.extend(subsegments)

    if len(segments) > 2:
        segments = segments[:5]

    # Przetwarzaj segmenty i łącz wyniki
    outputs = []
    for segment in segments:
        if len(segment) < 10 or len(segment) > 512:
            continue
        input_ids = tokenizer.encode(segment, return_tensors="pt")
        output = model(input_ids)[0]
        outputs.append(output)

    # Połącz wyniki segmentów
    combined_output = torch.cat(outputs, dim=1)
    return combined_output


def _pdf_from_url_to_text(pdf_url):
    # Pobierz plik PDF z podanego URL
    response = requests.get(pdf_url)

    if response.status_code != 200:
        return None

    # Przekształć zawartość w strumień plikowy
    pdf_content = BytesIO(response.content)

    # Stwórz PdfReader obiekt na podstawie strumienia
    pdf_reader = PyPDF2.PdfReader(pdf_content)

    # Inicjalizuj pusty string do przechowywania tekstu
    text = ''

    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    return text


def execute():
    os.environ[
        'ANTHROPIC_API_KEY'] = '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'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    categories = [
        'cs',
        'econ',
        'eess',
        'math',
        'physics',
        'q-bio',
        'q-fin',
        'stat'
    ]
    scraper = arxivscraper.Scraper(category='physics:cond-mat', date_from='2017-05-30', date_until='2017-06-01')
    output = scraper.scrape()

    df = pd.DataFrame(output)
    # limit df to 5 rows
    df['url'] = df['url'].replace('abs', 'pdf', regex=True)

    df = df.head(2)

    # Wczytaj model SciBERT
    model_name = "allenai/scibert_scivocab_uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    df["abstract_embedding"] = df["abstract"].apply(lambda x: _extract_features(x, tokenizer, model))

    # Zastosowanie z DataFrame
    df["pdf_text"] = df["url"].apply(_pdf_from_url_to_text)

    df["pdf_text_embedding"] = df["pdf_text"].apply(lambda x: _extract_features(x, tokenizer, model))

    # Wybierz kolumny
    df = df[['title', 'abstract', 'investors', 'business', 'pdf_text', 'abstract_embedding', 'pdf_text_embedding']]

    # Konwertuj typy danych
    df = df.astype({
        'title': 'string',
        'abstract': 'string',
        'investors': 'string',
        'business': 'string',
        'pdf_text': 'string'
    })

    df['abstract_embedding'] = df['abstract_embedding'].apply(lambda x: x.detach().cpu().numpy().tobytes())
    df['pdf_text_embedding'] = df['pdf_text_embedding'].apply(lambda x: x.detach().cpu().numpy().tobytes())

    # Utwórz połączenie z bazą danych SQLite
    conn = sqlite3.connect('example.db')

    # Użyj SQLAlchemy do zapisu ramki danych
    engine = create_engine('sqlite:///example.db')

    # Zdefiniuj typy danych dla kolumn
    dtype = {
        'title': types.String,
        'abstract': types.String,
        'investors': types.String,
        'business': types.String,
        'pdf_text': types.Text,
        'abstract_embedding': types.LargeBinary,
        'pdf_text_embedding': types.LargeBinary
    }

    # Zapisz ramkę danych do bazy SQLite
    df.to_sql('my_table', engine, if_exists='replace', index=False, dtype=dtype)

    # Zamknij połączenie
    conn.close()
