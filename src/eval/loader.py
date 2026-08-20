import json
import logging
from pathlib import Path
from typing import List
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EvalQuestion(BaseModel):
    id: str
    question: str
    gold_pmids: List[str]

def load_eval_questions(filepath: str = "tests/fixtures/eval_questions.json") -> List[EvalQuestion]:
    """Loads and parses evaluation questions from the JSON fixture file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found at {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = [EvalQuestion(**item) for item in data]
    logger.info(f"Loaded {len(questions)} evaluation questions from {filepath}")
    return questions