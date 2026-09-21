import os
import json
from typing import List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face InferenceClient
client = InferenceClient(token=HF_TOKEN)

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
RERANKER_MODEL = "parikshitup7/finetuned-medcpt-reranker"


def generate_embedding(text: str) -> List[float]:
    """Converts a chunk of text into a vector using HF InferenceClient."""
    # MedCPT max length is 512 tokens (~1,800 characters or ~380 words).
    # Truncate text if it exceeds 1,800 characters to prevent tensor shape errors.
    MAX_CHARS = 1800
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    response = client.feature_extraction(text, model=ARTICLE_MODEL)
    
    if hasattr(response, "tolist"):
        response = response.tolist()
        
    # Flatten nested single-item lists (e.g., [[...]] -> [...])
    while isinstance(response, list) and len(response) == 1 and isinstance(response[0], list):
        response = response[0]
        
    return response


def generate_query_embedding(query: str) -> List[float]:
    """Converts a user query into a vector using HF InferenceClient."""
    response = client.feature_extraction(query, model=QUERY_MODEL)
    
    if hasattr(response, "tolist"):
        return response.tolist()
    if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
        return response[0]
    return list(response)




def rerank_chunks(query: str, chunks: List[dict], top_n: int = 3) -> List[dict]:
    """Scores chunks using the Cross-Encoder model via HF InferenceClient."""
    if not chunks:
        return []

    payload = {
        "inputs": {
            "source_sentence": query,
            "sentences": [c.get("text", "") for c in chunks]
        }
    }

    try:
        response = client.post(json=payload, model=RERANKER_MODEL)
        scores = json.loads(response.decode("utf-8")) if isinstance(response, bytes) else response

        scored_chunks = []
        for i, chunk in enumerate(chunks):
            score = scores[i] if isinstance(scores, list) and i < len(scores) else 0.0
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = score
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return scored_chunks[:top_n]
    except Exception as e:
        print(f"Reranking fallback triggered: {e}")
        return chunks[:top_n]