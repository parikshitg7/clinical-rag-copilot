import pytest
from unittest.mock import patch
from src.eval.verifier_metrics import run_verifier_evaluation, is_hallucination
from src.schema import Label

def test_is_hallucination_logic():
    """Ensures our positive/negative mapping aligns with safety rules."""
    assert is_hallucination(Label.UNSUPPORTED) is True
    assert is_hallucination(Label.NOT_ENOUGH_INFO) is True
    assert is_hallucination(Label.SUPPORTED) is False

@patch("src.eval.verifier_metrics.verify_claim")
def test_run_verifier_evaluation_math(mock_verify, tmp_path):
    """Tests that Precision, Recall, and F1 are calculated correctly."""
    
    # We will mock the verifier to act perfectly. 
    # For our 5 items: 1 is Supported, 2 is Unsup, 3 is Not_Enough, 4 is Supp, 5 is Unsup.
    # A perfect verifier should return exactly those.
    mock_verify.side_effect = [
        Label.SUPPORTED,
        Label.UNSUPPORTED,
        Label.NOT_ENOUGH_INFO,
        Label.SUPPORTED,
        Label.UNSUPPORTED
    ]
    
    report_file = tmp_path / "mock_verifier_report.json"
    report = run_verifier_evaluation(output_file=str(report_file))
    
    # A perfect predictor should have 1.0 across the board
    assert report["metrics"]["precision"] == 1.0
    assert report["metrics"]["recall"] == 1.0
    assert report["metrics"]["f1_score"] == 1.0
    
    # 3 bad claims successfully caught, 2 good claims successfully passed
    assert report["confusion_matrix"]["true_positives_caught_hallucinations"] == 3
    assert report["confusion_matrix"]["true_negatives_good_claims_passed"] == 2