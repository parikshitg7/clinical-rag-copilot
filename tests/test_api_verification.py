# tests/test_api_verification.py
from fastapi.testclient import TestClient
from src.api import app
from src.schema import ClinicalAnswer, Claim
from src.verifier import Label

client = TestClient(app)

def test_ask_endpoint_filters_hallucinations(mocker):
    """
    Ensures that the /ask endpoint successfully runs the verification loop
    and filters out UNSUPPORTED claims before returning JSON to the user.
    """
    
    # 1. Mock the Generator to return one good claim and one hallucinated claim
    mocker.patch(
        "src.api.generate_clinical_answer",
        return_value=ClinicalAnswer(
            question="Does aspirin help?",
            claims=[
                Claim(text="Aspirin reduces heart attack risk.", source_id="doc1", source_chunk_id=1),
                Claim(text="Aspirin cures male pattern baldness.", source_id="doc1", source_chunk_id=1) # The Hallucination
            ],
            summary="Aspirin is analyzed.",
            verified=False
        )
    )
    
    # 2. Mock the retrieval system so it doesn't need to hit your real database
    mocker.patch("src.api.generate_query_embedding", return_value=[0.1]*768)
    mocker.patch("src.api.search_similar_chunks", return_value=[{"id": 1, "parent_doc_id": "doc1", "section": "Results", "text": "Aspirin reduces heart attack risk."}])
    mocker.patch("src.api.rerank_chunks", return_value=[{"id": 1, "parent_doc_id": "doc1", "section": "Results", "text": "Aspirin reduces heart attack risk.", "rerank_score": 0.99}])

    # 3. Mock the Verifier LLM call for a deterministic test
    def mock_verify(claim, source_text):
        if "baldness" in claim.text:
            return Label.UNSUPPORTED
        return Label.SUPPORTED
        
    mocker.patch("src.api.verify_claim", side_effect=mock_verify)

    # Execute the request
    response = client.get("/ask", params={"q": "Does aspirin help?"})
    
    assert response.status_code == 200
    data = response.json()
    
    # Assertions for Definition of Done
    assert data["verified"] is True
    assert len(data["claims"]) == 1  # The baldness claim should be missing!
    assert data["claims"][0]["text"] == "Aspirin reduces heart attack risk."