import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import instructor
from groq import Groq
from pydantic import BaseModel
import logging

from src.database import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class GeneratedEval(BaseModel):
    question: str

def get_random_chunks(limit: int = 100):
    """Pulls random chunks directly from Postgres to base our questions on."""
    conn = get_connection() 
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT parent_doc_id, text FROM chunks WHERE length(text) > 150 ORDER BY RANDOM() LIMIT %s", (limit,))
    results = cursor.fetchall()
    
    cursor.close()
    conn.close()
    return results

def generate_eval_dataset():
    logger.info("Connecting to Database to fetch 100 random chunks...")
    chunks = get_random_chunks(limit=100)
    
    client = instructor.from_groq(Groq(api_key=os.getenv("GROQ_API_KEY")))
    eval_dataset = []

    logger.info("Generating synthetic clinical questions via LLaMA-3...")
    for idx, chunk in enumerate(chunks, 1):
        prompt = f"""
        You are a medical professor writing a clinical exam. 
        Read the following medical text and write ONE specific, highly realistic clinical question that can be answered entirely by this text.
        Do not ask "What does this abstract say?". Ask a specific question about treatments, outcomes, or symptoms mentioned.
        
        Text: {chunk['text']}
        """
        
        try:
            result = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                response_model=GeneratedEval,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            eval_dataset.append({
                "id": f"eval_q_{idx}",
                "question": result.question,
                "gold_pmids": [chunk['parent_doc_id']]
            })
            logger.info(f"Generated Q{idx}/100: {result.question}")
            
        except Exception as e:
            logger.error(f"Error generating question {idx}: {e}")

    os.makedirs("tests/fixtures", exist_ok=True)
    with open("tests/fixtures/eval_questions.json", "w", encoding="utf-8") as f:
        json.dump(eval_dataset, f, indent=4)
        
    logger.info("Successfully generated 100 questions and saved to tests/fixtures/eval_questions.json!")

if __name__ == "__main__":
    generate_eval_dataset()