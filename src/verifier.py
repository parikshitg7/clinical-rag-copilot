# src/verifier.py
import os
from dotenv import load_dotenv
import instructor
from groq import Groq
from src.schema import Claim, VerificationResult, Label

# Load environment variables (ensure GROQ_API_KEY is in your .env)
load_dotenv()

# Initialize the Groq client and wrap it with Instructor
client = instructor.from_groq(Groq())

def verify_claim(claim: Claim, source_chunk_text: str) -> Label:
    """
    Judges whether a single claim is supported by the provided source chunk.
    """
    prompt = f"""
    You are a strict clinical entailment verifier. 
    Evaluate the provided Claim based ONLY on the provided Source Text.
    Do not use any outside medical knowledge.

    Source Text: {source_chunk_text}
    
    Claim: {claim.text}
    
    Rules for Classification:
    1. SUPPORTED: The claim is explicitly stated or logically entailed by the Source Text.
    2. UNSUPPORTED: The claim is directly contradicted by the Source Text.
    3. NOT_ENOUGH_INFO: The Source Text does not mention the topic, or lacks sufficient detail to prove or disprove the claim.

    Provide brief reasoning before outputting your label.
    """

    result = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_model=VerificationResult,
        messages=[
            {"role": "system", "content": "You are a highly strict clinical verification system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    
    return result.label