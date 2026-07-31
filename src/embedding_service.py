from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List

# The 3 MedCPT Models dictated by the Project Spec
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
RERANKER_MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

# Load the models into memory
article_model = SentenceTransformer(ARTICLE_MODEL_NAME)
query_model = SentenceTransformer(QUERY_MODEL_NAME)
cross_encoder = CrossEncoder(RERANKER_MODEL_NAME)

def generate_embedding(text: str) -> List[float]:
    """Converts a chunk of text into a vector."""
    embedding = article_model.encode(text)
    return embedding.tolist()

def generate_query_embedding(query: str) -> List[float]:
    """Converts a user's question into a vector."""
    embedding = query_model.encode(query)
    return embedding.tolist()

def rerank_chunks(query: str, chunks: List[dict], top_n: int = 3) -> List[dict]:
    """
    Takes a query and a list of retrieved chunks (from the database), 
    scores them using the MedCPT Cross-Encoder, and returns the top_n 
    chunks properly sorted by actual clinical relevance.
    """
    if not chunks:
        return []

    # The cross-encoder expects a list of pairs: [[query, chunk1], [query, chunk2], ...]
    pairs = [[query, chunk["text"]] for chunk in chunks]
    
    # Generate relevance scores for each pair
    scores = cross_encoder.predict(pairs)
    
    # Attach the new scores to our chunk dictionaries
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
        
    # Sort the chunks by the new rerank_score (highest score first)
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return reranked_chunks[:top_n]