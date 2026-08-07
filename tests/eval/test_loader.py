import pytest
from src.eval.loader import load_eval_questions, EvalQuestion

def test_load_eval_questions():
    """Tests that evaluation questions load and validate successfully into Pydantic models."""
    questions = load_eval_questions("tests/fixtures/eval_questions.json")
    
    assert isinstance(questions, list)
    assert len(questions) > 0
    assert isinstance(questions[0], EvalQuestion)
    assert questions[0].question_id == "eval_001"
    assert isinstance(questions[0].gold_source_ids, list)