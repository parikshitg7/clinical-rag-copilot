import os
import json
import time
import random
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from openai import OpenAI
import instructor
from pydantic import BaseModel
from src.database import get_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class GeneratedQuery(BaseModel):
    question: str

def get_chunks_for_training(limit: int = 250):
    """Pulls random chunks from Postgres to act as positive examples."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    # Grab chunks that have substantial text and aren't already used in eval
    cursor.execute("SELECT id, parent_doc_id, text FROM chunks WHERE length(text) > 150 ORDER BY RANDOM() LIMIT %s", (limit,))
    results = cursor.fetchall()
    
    # Grab a pool of random negatives
    cursor.execute("SELECT text FROM chunks WHERE length(text) > 150 ORDER BY RANDOM() LIMIT 1000")
    negatives = [row['text'] for row in cursor.fetchall()]
    conn.close()
    return results, negatives

def generate_triplets():
    logger.info("Fetching chunks for training data generation...")
    positives, negatives = get_chunks_for_training(limit=250)
    
    client = instructor.from_openai(
        OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
        ),
        mode=instructor.Mode.JSON
    )
    
    training_data = []

    logger.info("Generating training queries via LLM...")
    for idx, chunk in enumerate(positives, 1):
        prompt = f"""
        Read the following medical text and write ONE specific clinical question that this text answers perfectly.
        Do not ask "What does this abstract say?". Ask a specific medical question.
        Text: {chunk['text']}
        """
        try:
            result = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                response_model=GeneratedQuery,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            # Pick a random negative chunk that doesn't belong to the same abstract
            negative_text = random.choice(negatives)
            while negative_text == chunk['text']:
                negative_text = random.choice(negatives)
                
            training_data.append({
                "query": result.question,
                "positive": chunk['text'],
                "negative": negative_text
            })
            logger.info(f"Generated Triplet {idx}/250")
            time.sleep(2) # Rate limit protection
            
        except Exception as e:
            logger.error(f"Error generating triplet {idx}: {e}")

    os.makedirs("data", exist_ok=True)
    with open("data/training_triplets.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4)
        
    logger.info("Successfully generated training dataset at data/training_triplets.json!")

if __name__ == "__main__":
    generate_triplets()