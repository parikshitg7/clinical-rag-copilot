from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

# Import the exact function names from your embedding service
from src.embedding_service import generate_query_embedding, rerank_chunks
from src.repository import search_similar_chunks

app = FastAPI(
    title="Clinical Literature RAG Copilot",
    description="API for semantic search over clinical literature.",
    version="1.0.0"
)

# Define the structured output for our search endpoint
class SearchResult(BaseModel):
    id: int
    parent_doc_id: str
    section: str
    text: str
    score: float

@app.get("/search", response_model=List[SearchResult])
def search_clinical_literature(
    q: str = Query(..., description="The medical query to search for"),
    top_k: int = Query(50, description="Number of initial chunks to retrieve"),
    top_n: int = Query(8, description="Number of chunks to return after reranking")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        # 1. Embed the user's question using your exact function name
        query_vector = generate_query_embedding(q)
        
        # 2. Retrieve top-k chunks from Postgres
        retrieved_chunks = search_similar_chunks(query_vector, limit=top_k)
        
        if not retrieved_chunks:
            return []
            
        # 3. Rerank the chunks using the Cross-Encoder
        ranked_chunks = rerank_chunks(q, retrieved_chunks, top_n=top_n)
        
        # Format the output to match our Pydantic schema
        results = []
        # Your rerank_chunks returns a list of dicts containing a 'rerank_score'
        for chunk in ranked_chunks:
            results.append(
                SearchResult(
                    id=chunk['id'],
                    parent_doc_id=chunk['parent_doc_id'],
                    section=chunk['section'],
                    text=chunk['text'],
                    score=chunk['rerank_score']
                )
            )
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))