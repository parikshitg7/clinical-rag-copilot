import json
from pathlib import Path
from typing import List
from pydantic import BaseModel

class EvalQuestion(BaseModel):
    """Pydantic model representing a single evaluation question and its gold sources."""
    question_id: str
    question: str
    gold_source_ids: List[str]

def load_eval_questions(filepath: str | Path) -> List[EvalQuestion]:
    """Loads and validates evaluation questions from a JSON fixture."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Eval fixture not found at {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Pydantic automatically validates the JSON structure against the EvalQuestion model
    return [EvalQuestion(**item) for item in data]