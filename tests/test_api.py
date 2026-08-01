from fastapi.testclient import TestClient
from src.api import app
from unittest.mock import patch

client = TestClient(app)

# We mock the exact function names that src/api.py now imports
@patch("src.api.generate_query_embedding")
@patch("src.api.search_similar_chunks")
@patch("src.api.rerank_chunks")
def test_search_endpoint_success(mock_rerank, mock_search, mock_embed):
    # Setup mock returns
    mock_embed.return_value = [0.1] * 768
    
    mock_chunk = {
        "id": 1,
        "parent_doc_id": "PMID:12345",
        "section": "abstract",
        "text": "This is a mock medical text."
    }
    mock_search.return_value = [mock_chunk]
    
    # Your rerank_chunks returns a list of dictionaries with 'rerank_score'
    mock_reranked_chunk = mock_chunk.copy()
    mock_reranked_chunk["rerank_score"] = 0.95
    mock_rerank.return_value = [mock_reranked_chunk]

    # Execute the HTTP request
    response = client.get("/search?q=test%20query")
    
    # Assert Definition of Done: correctly-shaped responses
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["parent_doc_id"] == "PMID:12345"
    assert "score" in data[0]
    assert data[0]["score"] == 0.95

def test_search_endpoint_empty_query():
    # Execute with empty query
    response = client.get("/search?q=   ")
    
    # Assert validation catches it
    assert response.status_code == 400