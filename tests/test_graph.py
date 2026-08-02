# tests/test_graph.py
import pytest
from src.schema import Chunk, ClinicalAnswer, Claim, Label
from src.graph import clinical_graph

# Dummy data for the tests
dummy_chunks = [Chunk(parent_doc_id="123", section="Abstract", chunk_index=1, text="Aspirin reduces heart attack risk.")]

def test_graph_regenerate_then_succeed(mocker):
    """Proves the graph will retry an unsupported claim and succeed."""
    
    # Use functions to generate fresh objects so mutations in the graph don't ruin subsequent mock calls
    def generate_bad_answer(*args, **kwargs):
        return ClinicalAnswer(question="Test?", claims=[Claim(text="Bad claim", source_id="123", source_chunk_id=1)], summary="Sum", verified=False)
        
    def generate_good_answer(*args, **kwargs):
        return ClinicalAnswer(question="Test?", claims=[Claim(text="Good claim", source_id="123", source_chunk_id=1)], summary="Sum", verified=False)
    
    # 1. Mock the first generation to return a bad answer
    mocker.patch("src.graph.generate_clinical_answer", side_effect=generate_bad_answer)
    
    # 2. Mock the verifier to fail the first time, then pass the second time
    mocker.patch("src.graph.verify_claim", side_effect=[Label.UNSUPPORTED, Label.SUPPORTED])
    
    # 3. Mock the LLM inside the regenerate_node to return a good answer on retry
    mock_llm = mocker.patch("src.graph.client.chat.completions.create")
    mock_llm.side_effect = generate_good_answer
    
    # Run the graph
    result = clinical_graph.invoke({"question": "Test?", "chunks": dummy_chunks, "retries": 0})
    
    assert result["retries"] == 1  # It should have retried exactly once
    assert len(result["answer"].claims) == 1
    assert result["answer"].claims[0].text == "Good claim"
    assert result["answer"].verified == True


def test_graph_retry_cap_respected(mocker):
    """Proves the graph terminates after hitting the retry limit."""
    
    def generate_bad_answer(*args, **kwargs):
        return ClinicalAnswer(question="Test?", claims=[Claim(text="Always bad", source_id="123", source_chunk_id=1)], summary="Sum", verified=False)
    
    # Always generate fresh bad answers and always fail verification
    mocker.patch("src.graph.generate_clinical_answer", side_effect=generate_bad_answer)
    mocker.patch("src.graph.verify_claim", return_value=Label.UNSUPPORTED)
    
    mock_llm = mocker.patch("src.graph.client.chat.completions.create")
    mock_llm.side_effect = generate_bad_answer
    
    # Run the graph
    result = clinical_graph.invoke({"question": "Test?", "chunks": dummy_chunks, "retries": 0})
    
    assert result["retries"] == 2  # It should hit our cap of 2
    assert len(result["answer"].claims) == 0  # Bad claims are ultimately stripped before returning to user