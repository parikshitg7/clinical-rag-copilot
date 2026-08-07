import pytest
from unittest.mock import patch
from src.eval.retrieval_metrics import compute_metrics, run_retrieval_evaluation
from src.eval.loader import EvalQuestion

def test_compute_metrics_math():
    """Tests the exact mathematical formulas for Precision, Recall, and MRR."""
    gold_ids = ["123", "456"]
    
    # 5 items retrieved: The hit for '456' is at rank 2. The hit for '123' is at rank 4.
    retrieved_ids = ["999", "456", "111", "123", "000"]
    
    metrics = compute_metrics(gold_ids, retrieved_ids, k=5)
    
    # Precision@5 = 2 relevant / 5 retrieved = 0.4
    assert metrics["precision"] == 0.4
    # Recall@5 = 2 unique relevant found / 2 total relevant = 1.0
    assert metrics["recall"] == 1.0
    # MRR = First hit is at index 1 (rank 2), so 1 / 2 = 0.5
    assert metrics["mrr"] == 0.5

@patch("src.eval.retrieval_metrics.load_eval_questions")
@patch("src.eval.retrieval_metrics.generate_query_embedding")
@patch("src.eval.retrieval_metrics.search_similar_chunks")
@patch("src.eval.retrieval_metrics.rerank_chunks")
def test_run_retrieval_evaluation(mock_rerank, mock_search, mock_embed, mock_load, tmp_path):
    """Ensures the evaluation pipeline runs and generates a JSON report without hitting the live DB."""
    # Mock data setup
    mock_load.return_value = [
        EvalQuestion(question_id="1", question="test?", gold_source_ids=["123"])
    ]
    mock_embed.return_value = [0.1] * 768
    mock_search.return_value = [{"parent_doc_id": "123", "text": "text chunk"}]
    mock_rerank.return_value = [{"parent_doc_id": "123", "text": "text chunk", "rerank_score": 0.99}]
    
    # Use Pytest's temporary directory for the report output
    report_file = tmp_path / "test_report.json"
    
    report = run_retrieval_evaluation(initial_k=1, eval_k=1, output_file=str(report_file))
    
    assert "baseline_metrics" in report
    assert "reranked_metrics" in report
    assert report_file.exists()