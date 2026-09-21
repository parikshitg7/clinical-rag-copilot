from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

# Import the LangGraph state machine
from src.graph import clinical_graph

# Import function names from your embedding service and repository
from src.embedding_service import generate_query_embedding, rerank_chunks
from src.repository import search_similar_chunks

# Imports for Generation and Verification schemas
from src.schema import Chunk, ClinicalAnswer

app = FastAPI(
    title="Clinical Literature RAG Copilot",
    description="API for semantic search over clinical literature.",
    version="1.0.0"
)

# Helper function to construct PubMed URL
def build_pubmed_url(parent_doc_id: str) -> str:
    pmid = str(parent_doc_id).replace("PMID_", "").strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

# Updated SearchResult schema with url field
class SearchResult(BaseModel):
    id: int
    parent_doc_id: str
    section: str
    text: str
    score: float
    url: str  # Direct clickable link to PubMed

@app.get("/")
def read_root():
    return {"status": "online", "message": "API is running. Go to /docs for Swagger UI."}

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
            doc_id = chunk.get('parent_doc_id', '')
            score = chunk.get('rerank_score', chunk.get('similarity', 0.0))
            
            results.append(
                SearchResult(
                    id=chunk['id'],
                    parent_doc_id=doc_id,
                    section=chunk.get('section') or '',
                    text=chunk['text'],
                    score=float(score),
                    url=build_pubmed_url(doc_id)
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
        
        chunk_objects = []
        for idx, c in enumerate(ranked_chunks):
            chunk_objects.append(
                Chunk(
                    parent_doc_id=c.get('parent_doc_id', ''),
                    section=c.get('section') or '',
                    chunk_index=idx,
                    text=c['text']
                )
            )

        initial_state = {
            "question": q,
            "chunks": chunk_objects,
        }
        
        final_state = clinical_graph.invoke(initial_state)
        return final_state["answer"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))