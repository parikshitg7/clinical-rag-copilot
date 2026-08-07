import pytest
from unittest.mock import patch, MagicMock
from src.eval.e2e_metrics import evaluate_answer_claims, run_e2e_evaluation
from src.schema import ClinicalAnswer, Claim, Chunk, Label

def test_evaluate_answer_claims():
    """Tests that claims are correctly counted and verified."""
    chunk = Chunk(parent_doc_id="123", chunk_index=0, text="Test text", section="A")
    claim1 = Claim(text="Good claim", source_id="123", source_chunk_id=0)
    claim2 = Claim(text="Bad claim", source_id="123", source_chunk_id=0)
    
    answer = ClinicalAnswer(question="Q", claims=[claim1, claim2], summary="Sum", verified=False)
    
    with patch("src.eval.e2e_metrics.verify_claim") as mock_verify:
        # Mock first claim as SUPPORTED, second as UNSUPPORTED
        mock_verify.side_effect = [Label.SUPPORTED, Label.UNSUPPORTED]
        
        total, unsup = evaluate_answer_claims(answer, [chunk])
        
        assert total == 2
        assert unsup == 1

@patch("src.eval.e2e_metrics.search_similar_chunks")
@patch("src.eval.e2e_metrics.generate_query_embedding")
@patch("src.eval.e2e_metrics.load_eval_questions")
@patch("src.eval.e2e_metrics.clinical_graph.invoke")
@patch("src.eval.e2e_metrics.generate_clinical_answer")
@patch("src.eval.e2e_metrics.evaluate_answer_claims")
def test_run_e2e_evaluation(mock_eval, mock_gen, mock_invoke, mock_load, mock_embed, mock_search, tmp_path):
    """Tests the E2E script correctly aggregates metrics and outputs a report."""
    
    # Setup simple mocks to pass the loops
    mock_load.return_value = [MagicMock(question="test?")]
    mock_embed.return_value = [0.1]
    mock_search.return_value = [{"parent_doc_id": "1", "chunk_index": 1, "text": "text"}]
    
    # Mock the answers
    mock_gen.return_value = ClinicalAnswer(question="Q", claims=[], summary="", verified=False)
    mock_invoke.return_value = {"answer": ClinicalAnswer(question="Q", claims=[], summary="", verified=True)}
    
    # Mock the metric math: Baseline has 1 bad out of 2. Graph has 0 bad out of 1.
    mock_eval.side_effect = [
        (2, 1), # Baseline returns: 2 total, 1 unsupported
        (1, 0)  # Graph returns: 1 total, 0 unsupported
    ]
    
    report_file = tmp_path / "e2e_test.json"
    report = run_e2e_evaluation(output_file=str(report_file))
    
    assert report["baseline_rag"]["total_claims_made"] == 2
    assert report["baseline_rag"]["hallucination_rate"] == 0.5
    assert report["verified_langgraph"]["total_claims_made"] == 1
    assert report["verified_langgraph"]["hallucination_rate"] == 0.0
    assert report["improvement"]["absolute_reduction"] == 0.5