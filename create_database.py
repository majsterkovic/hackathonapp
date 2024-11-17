from tqdm import tqdm
import logging

import arxivscraper
import anthropic
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import PyPDF2
from io import BytesIO
import sqlite3
from sqlalchemy import create_engine, types
import yake


# def _extract_features(text, tokenizer, model):

#     if text is None:
#         return None
    
#     segment_length = 512
#     segments = []
#     for i in range(0, len(text), segment_length):
#         segment = text[i:i + segment_length]
#         if len(segment.split()) <= 512:
#             segments.append(segment)
#         else:
#             # Podziel dłuższy segment na mniejsze
#             subsegments = [" ".join(segment.split()[j:j + 512]) for j in range(0, len(segment.split()), 512)]
#             segments.extend(subsegments)

#     if len(segments) > 2:
#         segments = segments[:5]

#     # Przetwarzaj segmenty i łącz wyniki
#     outputs = []
#     for segment in segments:
#         if len(segment) < 10 or len(segment) > 512:
#             continue
#         input_ids = tokenizer.encode(segment, return_tensors="pt")
#         output = model(input_ids)[0]
#         outputs.append(output)

    # Połącz wyniki segmentów
def _extract_features(text, tokenizer, model):

    if text is None:
        return None
    
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
    segment_embeddings = []
    for segment in segments:
        if len(segment) < 10 or len(segment) > 512:
            continue
        input_ids = tokenizer.encode(segment, return_tensors="pt")
        output = model(input_ids)[0]  # [batch_size, seq_length, hidden_size]
        
        # Weź średnią po długości sekwencji (wymiar 1)
        segment_embedding = torch.mean(output, dim=1)  # [batch_size, hidden_size]
        segment_embeddings.append(segment_embedding)

    if not segment_embeddings:
        return None

    # Metoda 1: Średnia ze wszystkich segmentów
    document_embedding = torch.mean(torch.stack(segment_embeddings, dim=0), dim=0)
    return document_embedding


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


def _rewrite_abstract(abstract: str, prompt: str) -> str:
    """
    Rewrites an academic abstract to be more investor-friendly using Claude AI.

    Args:
        abstract (str): The academic abstract to rewrite

    Returns:
        str: The investor-friendly version of the abstract
    """

    # Initialize Anthropic client
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY")
    )

    # Generate response
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="You are a creative writing assistant.",
        messages=[
            {"role": "user", "content": f"Hello Claude. {prompt}"},
            {"role": "user", "content": abstract},
            {"role": "assistant", "content": "Here is a paraphrased version of the abstract:"}
        ]
    )

    return message.content

def _extract_keywords(text):
    language = "en"
    numOfKeywords = 4

    custom_kw_extractor = yake.KeywordExtractor(
        lan=language,
        top=numOfKeywords,
        features=None
        )
    
    keywords = custom_kw_extractor.extract_keywords(text)
    keywords = [keyword for keyword, score in keywords]
    keywords = ', '.join(keywords)
    return keywords

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
    scraper = arxivscraper.Scraper(
        category='cs',
        date_from='2020-05-30',
        date_until='2020-06-05'
        )
    output = scraper.scrape()

    df = pd.DataFrame(output)
    # limit df to 5 rows
    df['url'] = df['url'].replace('abs', 'pdf', regex=True)

    df = df.head(10)
    tqdm.pandas(desc="Processing pdf_text")
    df["pdf_text"] = df["url"].progress_apply(_pdf_from_url_to_text)
    df = df.dropna(subset=["pdf_text"])
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    df['emails'] = df["pdf_text"].apply(lambda x: re.findall(email_regex, x))
    df = df[df['emails'].apply(len) > 0]

    logging.info("Done emails")
    df = df.head(10)

    prompt_for_investors = "Please shortly rewrite this abstract in a way that highlights the potential business opportunities and market impact of the described approach."
    prompt_for_business = "Please shortly rewrite this abstract to emphasize the practical applications, product development potential, and competitive advantages of the described technical approach."

    tqdm.pandas(desc="Processing business")
    df["business"] = df.progress_apply(lambda x: _rewrite_abstract(x['abstract'], prompt_for_business)[0].text, axis=1)

    tqdm.pandas(desc="Processing investors")
    df["investors"] = df.progress_apply(lambda x: _rewrite_abstract(x['abstract'], prompt_for_investors)[0].text, axis=1)

    # Wczytaj model SciBERT
    model_name = "allenai/scibert_scivocab_uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tqdm.pandas(desc="Processing abstract_embedding")
    df["abstract_embedding"] = df["abstract"].progress_apply(lambda x: _extract_features(x, tokenizer, model))

    # Zastosowanie z DataFrame
    tqdm.pandas(desc="Processing pdf_text")
    df["pdf_text"] = df["url"].progress_apply(_pdf_from_url_to_text)

    tqdm.pandas(desc="Processing pdf_text_embedding")
    df["pdf_text_embedding"] = df["pdf_text"].progress_apply(lambda x: _extract_features(x, tokenizer, model))

    tqdm.pandas(desc="Processing keywords")
    df['keywords'] = df['pdf_text'].progress_apply(lambda x: _extract_keywords(x))

    # Wybierz kolumny
    df = df[['title', 'abstract', 'investors', 'business', 'pdf_text', 'abstract_embedding', 'pdf_text_embedding', 'authors', 'url', 'keywords', 'emails']]

    # Konwertuj typy danych
    df = df.astype({
        'title': 'string',
        'abstract': 'string',
        'investors': 'string',
        'business': 'string',
        'pdf_text': 'string',
        'authors': 'string',
        'emails': 'string',
        'url': 'string'
    })

    tqdm.pandas(desc="Processing abstract_embedding")
    df['abstract_embedding'] = df['abstract_embedding'].progress_apply(lambda x: x.detach().cpu().numpy().tobytes() if x is not None else None)

    tqdm.pandas(desc="Processing pdf_text_embedding")
    df['pdf_text_embedding'] = df['pdf_text_embedding'].progress_apply(lambda x: x.detach().cpu().numpy().tobytes() if x is not None else None)

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
        'keywords': types.String,
        'authors': types.String,
        'emails': types.String,
        'abstract_embedding': types.LargeBinary,
        'pdf_text_embedding': types.LargeBinary
    }

    # Zapisz ramkę danych do bazy SQLite
    df.to_sql('my_table', engine, if_exists='replace', index=False, dtype=dtype)

    # Zamknij połączenie
    conn.close()

tqdm.pandas()
execute()