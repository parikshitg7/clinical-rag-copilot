# tests/test_api_integration.py
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_ask_endpoint_integration(mocker):
    """Verifies that the /ask endpoint triggers the graph successfully."""
    
    # 1. Mock the pipeline dependencies so they don't hit the real database/embedding models
    mocker.patch("src.api.generate_query_embedding", return_value=[0.1] * 768)
    mocker.patch("src.api.search_similar_chunks", return_value=[{'id': 1, 'parent_doc_id': '123', 'section': 'sec', 'text': 'text'}])
    mocker.patch("src.api.rerank_chunks", return_value=[{'id': 1, 'parent_doc_id': '123', 'section': 'sec', 'text': 'text', 'rerank_score': 0.9}])
    
    # 2. Mock the graph itself so we don't spend Groq API credits during tests
    mock_graph = mocker.patch("src.api.clinical_graph")
    
    # Return a fake successful graph state
    from src.schema import ClinicalAnswer
    mock_graph.invoke.return_value = {
        "answer": ClinicalAnswer(
            question="Is Aspirin good?", 
            claims=[], 
            summary="Yes.", 
            verified=True
        )
    }
    
    # 3. Call the API using the TestClient
    response = client.get("/ask?q=Is+Aspirin+good?")
    
    # 4. Assertions
    assert response.status_code == 200
    assert response.json()["question"] == "Is Aspirin good?"
    assert response.json()["verified"] is True