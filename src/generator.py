import os
from typing import List
from groq import Groq
import instructor
from dotenv import load_dotenv  # <-- 1. Import this
from src.schema import Chunk, ClinicalAnswer

load_dotenv()  # <-- 2. Call this BEFORE initializing the client

# Initialize the Groq client and patch it with Instructor to enforce Pydantic schemas
client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY", "dummy-key-for-tests")), # <-- Added fallback for CI/CD
    mode=instructor.Mode.JSON
)

SYSTEM_PROMPT = """You are a strictly fact-based clinical AI assistant.
You will be provided with a user's clinical question and a set of retrieved document chunks.
Your task is to answer the user's question using ONLY the provided chunks.

Rules:
1. Extract specific, atomic claims from the text that answer the question.
2. For each claim, you MUST cite the exact source_id and source_chunk_id it came from.
3. Do NOT hallucinate external knowledge. 
4. If the provided chunks cannot answer the question, state that clearly in your summary and return an empty list of claims.
"""

def generate_clinical_answer(question: str, chunks: List[Chunk]) -> ClinicalAnswer:
    """
    Sends the question and retrieved chunks to Groq and returns a structured ClinicalAnswer.
    """
    # Format the retrieved chunks into the prompt context
    # Format the retrieved chunks into the prompt context
    context_text = "\n\n".join(
        [f"[Chunk ID: {c.chunk_index} | Source: {c.parent_doc_id}]\n{c.text}" for c in chunks]
    )
    
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}"

    # Instructor handles the JSON schema enforcement automatically
    answer = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Fast and highly capable of strict instruction following
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_model=ClinicalAnswer,
        temperature=0.0,  # Zero creativity, strictly factual extraction
    )

    return answer