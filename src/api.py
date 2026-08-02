from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List

# Import the exact function names from your embedding service
from src.embedding_service import generate_query_embedding, rerank_chunks
from src.repository import search_similar_chunks

# Imports for Generation and Verification
from src.schema import Chunk, ClinicalAnswer
from src.generator import generate_clinical_answer
from src.verifier import verify_claim, Label

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
            return ClinicalAnswer(question=q, claims=[], summary="No relevant medical literature found to answer this question.", verified=True)
            
        ranked_chunks = rerank_chunks(q, retrieved_chunks, top_n=top_n)
        
        # 2. Convert raw DB dicts into Chunk objects for the Generator
        chunk_objects = []
        for c in ranked_chunks:
            # We use the DB 'id' as the chunk_index so the LLM cites the exact DB row
            chunk_index = c.get('chunk_index', c['id']) 
            chunk_objects.append(
                Chunk(
                    parent_doc_id=c['parent_doc_id'],
                    section=c['section'],
                    chunk_index=chunk_index,
                    text=c['text']
                )
            )

        # 3. Generate the Initial Answer
        initial_answer = generate_clinical_answer(q, chunk_objects)

        # 4. Verify Claims (Phase 6.2 Integration)
        verified_claims = []
        for claim in initial_answer.claims:
            # Find the exact source text the LLM cited
            source_text = ""
            for c in chunk_objects:
                if c.parent_doc_id == claim.source_id and c.chunk_index == claim.source_chunk_id:
                    source_text = c.text
                    break
            
            # If the LLM hallucinated a non-existent chunk ID, we drop it immediately
            if not source_text:
                continue

            # Run it through the strict clinical verifier
            label = verify_claim(claim, source_text)
            
            # Keep only the claims that are NOT labeled as UNSUPPORTED
            if label != Label.UNSUPPORTED:
                verified_claims.append(claim)

        # 5. Update and return the verified answer
        initial_answer.claims = verified_claims
        initial_answer.verified = True

        return initial_answer

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))