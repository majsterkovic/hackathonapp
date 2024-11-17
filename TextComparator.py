import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class TextComparator:
      def __init__(self):
            # Inicjalizacja modelu do generowania embeddingów
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.database_embeddings = None
            self.database_texts = None
