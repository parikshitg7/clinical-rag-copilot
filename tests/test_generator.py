import pytest
from unittest.mock import patch
from src.generator import generate_clinical_answer
from src.schema import Chunk, ClinicalAnswer, Claim

@patch("src.generator.client.chat.completions.create")
def test_generate_clinical_answer_success(mock_create):
    """Tests the generation logic without hitting the actual Groq API."""
    
    # 1. Setup the mock response to match our new ID logic
    mock_create.return_value = ClinicalAnswer(
        question="Is Metformin safe?",
        claims=[
            Claim(
                text="Metformin is generally safe and well-tolerated.", 
                source_id="doc_001",      # Matches parent_doc_id
                source_chunk_id=0         # Matches chunk_index
            )
        ],
        summary="Metformin is deemed safe according to the provided text.",
        verified=False
    )

    # 2. Setup the fake input data using ONLY valid Chunk fields
    fake_chunks = [
        Chunk(
            parent_doc_id="doc_001",
            section="results",
            chunk_index=0,
            text="Clinical trials show Metformin is generally safe and well-tolerated."
        )
    ]
    
    # 3. Execute the function
    answer = generate_clinical_answer("Is Metformin safe?", fake_chunks)
    
    # 4. Assertions
    assert answer.question == "Is Metformin safe?"
    assert len(answer.claims) == 1
    assert answer.claims[0].source_chunk_id == 0  # Updated assertion
    
    # Verify Groq/Instructor was called correctly
    mock_create.assert_called_once()
    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "llama3-70b-8192"
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_model"] == ClinicalAnswer