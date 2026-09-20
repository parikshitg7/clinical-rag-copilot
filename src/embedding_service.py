import os
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

ARTICLE_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/ncbi/MedCPT-Query-Encoder"

# Pointing to your fine-tuned reranker on Hugging Face Hub
RERANKER_MODEL_URL = "https://api-inference.huggingface.co/models/parikshitup7/finetuned-medcpt-reranker"

def _get_hf_embedding(url: str, text: str) -> List[float]:
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    response = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    response.raise_for_status()
    res = response.json()
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        return res[0]
    return res

def generate_embedding(text: str) -> List[float]:
    """Converts a chunk of text into a vector via HF Inference API."""
    return _get_hf_embedding(ARTICLE_MODEL_URL, text)

def generate_query_embedding(query: str) -> List[float]:
    """Converts a user's question into a vector via HF Inference API."""
    return _get_hf_embedding(QUERY_MODEL_URL, query)

def rerank_chunks(query: str, chunks: List[dict], top_n: int = 3) -> List[dict]:
    """Scores chunks using your fine-tuned Cross-Encoder API and returns top_n ordered by score."""
    if not chunks:
        return []

    pairs = [{"text": query, "text_pair": chunk["text"]} for chunk in chunks]
    
    try:
        response = requests.post(
            RERANKER_MODEL_URL,
            headers=HEADERS,
            json={"inputs": pairs, "options": {"wait_for_model": True}},
            timeout=30
        )
        response.raise_for_status()
        scores_data = response.json()
        
        for i, chunk in enumerate(chunks):
            if isinstance(scores_data, list) and i < len(scores_data):
                item = scores_data[i]
                if isinstance(item, list) and len(item) > 0 and "score" in item[0]:
                    chunk["rerank_score"] = float(item[0]["score"])
                elif isinstance(item, dict) and "score" in item:
                    chunk["rerank_score"] = float(item["score"])
                elif isinstance(item, (int, float)):
                    chunk["rerank_score"] = float(item)
                else:
                    chunk["rerank_score"] = 0.0
            else:
                chunk["rerank_score"] = 0.0

    except Exception as e:
        print(f"Reranking API note: {e}. Preserving default vector ordering.")
        for chunk in chunks:
            chunk.setdefault("rerank_score", 0.0)

    reranked_chunks = sorted(chunks, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked_chunks[:top_n]