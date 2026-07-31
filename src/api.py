from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from src.embedding_service import generate_query_embedding
from src.repository import search_similar_chunks
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the web application
app = FastAPI(
    title="Clinical Literature Vector Search API",
    description="An API to semantically search medical text using MedCPT embeddings and pgvector.",
    version="1.0.0"
)

# Define the expected JSON structure for incoming requests
class SearchRequest(BaseModel):
    query: str
    limit: int = 5

# Define the expected JSON structure for outgoing responses
class SearchResult(BaseModel):
    id: int
    parent_doc_id: str
    section: str
    text: str
    similarity: float

@app.post("/search", response_model=List[SearchResult])
def search_clinical_text(request: SearchRequest):
    """
    Takes a natural language medical question, converts it to a vector, 
    and returns the most mathematically similar chunks from the database.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")
    
    logger.info(f"Received search request for query: '{request.query}'")
    
    try:
        # 1. Convert the query string into a 768-dimension vector
        query_vector = generate_query_embedding(request.query)
        
        # 2. Search the Postgres database
        results = search_similar_chunks(query_vector, limit=request.limit)
        
        # 3. Return the results (FastAPI automatically converts this list of dicts to JSON)
        return results
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during vector search.")

@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "healthy", "service": "clinical-rag-search-api"}