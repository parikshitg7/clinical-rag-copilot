import os
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List

# MedCPT Models
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
BASELINE_RERANKER_NAME = "ncbi/MedCPT-Cross-Encoder"

# Local path for fine-tuned reranker
FINETUNED_RERANKER_PATH = os.path.join("models", "finetuned_medcpt_reranker")

# Dynamically switch model path based on fine-tuned model existence
if os.path.exists(FINETUNED_RERANKER_PATH):
    RERANKER_MODEL_NAME = FINETUNED_RERANKER_PATH
    print(f"Loading Fine-Tuned Reranker from: {FINETUNED_RERANKER_PATH}")
else:
    RERANKER_MODEL_NAME = BASELINE_RERANKER_NAME
    print(f"Loading Baseline Reranker: {BASELINE_RERANKER_NAME}")

# Load models
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
    """Scores chunks using the Cross-Encoder and returns top_n ordered by score."""
    if not chunks:
        return []

    pairs = [[query, chunk["text"]] for chunk in chunks]
    scores = cross_encoder.predict(pairs)
    
    for i, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[i])
        
    reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    return reranked_chunks[:top_n]