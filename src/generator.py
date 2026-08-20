import os
from typing import List
from openai import OpenAI
import instructor
from dotenv import load_dotenv

from src.schema import Chunk, ClinicalAnswer

load_dotenv()

# Initialize OpenAI client pointed to Groq with instructor
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "dummy-key-for-tests"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    ),
    mode=instructor.Mode.JSON
)

SYSTEM_PROMPT = """You are a strictly fact-based clinical AI assistant.
You will be provided with a user's clinical question and a set of retrieved document chunks.
Your task is to answer the user's question using ONLY the provided chunks.

Rules:
1. Extract specific, atomic claims from the text that answer the question.
2. For each claim in `claims`, you MUST provide flat properties: `text` (string), `source_id` (string), and `source_chunk_id` (integer). Do NOT create a nested "source" object.
3. Do NOT hallucinate external knowledge. 
4. If the provided chunks cannot answer the question, state that clearly in your summary and return an empty list of claims.
"""

def generate_clinical_answer(question: str, chunks: List[Chunk]) -> ClinicalAnswer:
    """
    Sends the question and retrieved chunks to the LLM and returns a structured ClinicalAnswer.
    """
    context_text = "\n\n".join(
        [f"[Chunk ID: {c.chunk_index} | Source: {c.parent_doc_id}]\n{c.text}" for c in chunks]
    )
    
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}"

    answer = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_model=ClinicalAnswer,
        temperature=0.0,
    )

    return answer