import os
from typing import List
from sentence_transformers import CrossEncoder, SentenceTransformer

# MedCPT Models
ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"

# Point directly to your Hugging Face repository
RERANKER_MODEL_NAME = "parikshitup7/finetuned-medcpt-reranker"

print(f"Loading Fine-Tuned Reranker from Hugging Face: {RERANKER_MODEL_NAME}")

# Load models from Hugging Face Hub
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


def rerank_chunks(
    query: str, chunks: List[dict], top_n: int = 3
) -> List[dict]:
  """Scores chunks using the Cross-Encoder and returns top_n ordered by score."""
  if not chunks:
    return []

  pairs = [[query, chunk["text"]] for chunk in chunks]
  scores = cross_encoder.predict(pairs)

  for i, chunk in enumerate(chunks):
    chunk["rerank_score"] = float(scores[i])

  reranked_chunks = sorted(
      chunks, key=lambda x: x["rerank_score"], reverse=True
  )
  return reranked_chunks[:top_n]