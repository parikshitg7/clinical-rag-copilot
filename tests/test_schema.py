import pytest
from pydantic import ValidationError
from src.schema import Claim, ClinicalAnswer

def test_valid_clinical_answer():
    """Tests that valid data is successfully parsed into the models."""
    claim = Claim(
        text="Metformin is effective for lowering blood sugar.",
        source_id="PMID:12345",
        source_chunk_id=42
    )
    
    answer = ClinicalAnswer(
        question="What does Metformin do?",
        claims=[claim],
        summary="Metformin lowers blood sugar.",
        verified=False
    )
    
    assert answer.question == "What does Metformin do?"
    assert len(answer.claims) == 1
    assert answer.claims[0].source_id == "PMID:12345"
    assert answer.verified is False

def test_invalid_claim_missing_fields():
    """Tests that missing required fields trigger a validation error."""
    with pytest.raises(ValidationError):
        # Missing source_id and source_chunk_id
        Claim(text="This claim has no proof.")

def test_invalid_clinical_answer_missing_claims():
    """Tests that a clinical answer requires the claims list."""
    with pytest.raises(ValidationError):
        ClinicalAnswer(
            question="Where are the claims?",
            summary="There are none."
            # Missing the required 'claims' list
        )