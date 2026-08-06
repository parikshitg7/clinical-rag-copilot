import json
from pathlib import Path
from src.eval.loader import load_eval_questions, EvalQuestion

def test_load_eval_questions(tmp_path: Path):
    # Setup: Create a temporary JSON fixture using pytest's tmp_path
    fixture_file = tmp_path / "test_eval_questions.json"
    sample_data = [
        {
            "question_id": "test_1",
            "question": "Is this a test question?",
            "gold_source_ids": ["12345", "67890"]
        }
    ]
    fixture_file.write_text(json.dumps(sample_data))
    
    # Execution: Load the data
    questions = load_eval_questions(fixture_file)
    
    # Assertion: Verify Pydantic parsed it correctly
    assert len(questions) == 1
    assert isinstance(questions[0], EvalQuestion)
    assert questions[0].question_id == "test_1"
    assert questions[0].question == "Is this a test question?"
    assert questions[0].gold_source_ids == ["12345", "67890"]