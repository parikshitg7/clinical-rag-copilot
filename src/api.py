# src/api.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

# Import the LangGraph state machine
from src.graph import clinical_graph

# Import the exact function names from your embedding service
from src.embedding_service import generate_query_embedding, rerank_chunks
from src.repository import search_similar_chunks

# Imports for Generation and Verification schemas
from src.schema import Chunk, ClinicalAnswer

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
        query_vector = generate_query_embedding(q)
        retrieved_chunks = search_similar_chunks(query_vector, limit=top_k)
        
        if not retrieved_chunks:
            return []
            
        ranked_chunks = rerank_chunks(q, retrieved_chunks, top_n=top_n)
        
        results = []
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


@app.get("/ask", response_model=ClinicalAnswer)
def ask_clinical_question(
    q: str = Query(..., description="The medical query to ask the AI"),
    top_k: int = Query(50, description="Number of initial chunks to retrieve"),
    top_n: int = Query(5, description="Number of chunks to send to the AI")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # 1. Retrieve & Rerank Context
        query_vector = generate_query_embedding(q)
        retrieved_chunks = search_similar_chunks(query_vector, limit=top_k)
        
        if not retrieved_chunks:
            return ClinicalAnswer(
                question=q, 
                claims=[], 
                summary="No relevant medical literature found to answer this question.", 
                verified=True
            )
            
        ranked_chunks = rerank_chunks(q, retrieved_chunks, top_n=top_n)
        
        # 2. Convert raw DB dicts into Chunk objects for the Graph
        chunk_objects = []
        for c in ranked_chunks:
            chunk_index = c.get('chunk_index', c['id']) 
            chunk_objects.append(
                Chunk(
                    parent_doc_id=c['parent_doc_id'],
                    section=c['section'],
                    chunk_index=chunk_index,
                    text=c['text']
                )
            )

        # 3. Hand off to the LangGraph Orchestrator
        # This completely replaces the manual loop!
        initial_state = {
            "question": q,
            "chunks": chunk_objects,
        }
        
        # Invoke the graph
        final_state = clinical_graph.invoke(initial_state)
        
        return final_state["answer"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))