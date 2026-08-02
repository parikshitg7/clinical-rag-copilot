import pytest
from src.schema import Claim, Label
from src.verifier import verify_claim

def test_verify_claim_supported():
    claim = Claim(text="Aspirin reduces the risk of myocardial infarction.", source_id="123", source_chunk_id=1)
    source_text = "In a recent double-blind study, daily aspirin administration was shown to significantly reduce the risk of myocardial infarction."
    
    result = verify_claim(claim, source_text)
    assert result == Label.SUPPORTED

def test_verify_claim_unsupported():
    claim = Claim(text="Aspirin increases the risk of myocardial infarction.", source_id="123", source_chunk_id=1)
    source_text = "In a recent double-blind study, daily aspirin administration was shown to significantly reduce the risk of myocardial infarction."
    
    result = verify_claim(claim, source_text)
    assert result == Label.UNSUPPORTED

def test_verify_claim_not_enough_info():
    claim = Claim(text="Aspirin causes severe liver damage in children.", source_id="123", source_chunk_id=1)
    source_text = "In a recent double-blind study, daily aspirin administration was shown to significantly reduce the risk of myocardial infarction."
    
    result = verify_claim(claim, source_text)
    assert result == Label.NOT_ENOUGH_INFO