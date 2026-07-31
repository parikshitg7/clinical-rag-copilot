from sentence_transformers import SentenceTransformer
from typing import List

# MedCPT uses two separate models: one for articles, one for user queries
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

# Load both models into memory
article_model = SentenceTransformer(ARTICLE_MODEL_NAME)
query_model = SentenceTransformer(QUERY_MODEL_NAME)

def generate_embedding(text: str) -> List[float]:
    """
    Converts a chunk of text from the database into a vector 
    using the MedCPT Article Encoder.
    """
    embedding = article_model.encode(text)
    return embedding.tolist()

def generate_query_embedding(query: str) -> List[float]:
    """
    Converts a user's question into a vector 
    using the MedCPT Query Encoder.
    """
    embedding = query_model.encode(query)
    return embedding.tolist()