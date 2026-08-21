import os
from dotenv import load_dotenv
import instructor
from openai import OpenAI
from src.schema import Claim, VerificationResult, Label

load_dotenv()

# Initialize OpenAI client pointed to Groq with instructor JSON mode
client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY", "dummy-key-for-tests"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    ),
    mode=instructor.Mode.JSON
)

def verify_claim(claim: Claim, source_chunk_text: str) -> Label:
    """
    Judges whether a single claim is supported by the provided source chunk using gpt-oss-120b.
    """
    prompt = f"""
    You are a strict clinical entailment verifier. 
    Evaluate the provided Claim based ONLY on the provided Source Text.
    Do not use any outside medical knowledge.

    Source Text: {source_chunk_text}
    
    Claim: {claim.text}
    
    Rules for Classification (Output exact lowercase strings):
    1. "supported": The claim is explicitly stated or logically entailed by the Source Text.
    2. "unsupported": The claim is directly contradicted by the Source Text.
    3. "not_enough_info": The Source Text does not mention the topic, or lacks sufficient detail to prove or disprove the claim.

    Provide brief reasoning before outputting your label.
    """

    result = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_model=VerificationResult,
        messages=[
            {"role": "system", "content": "You are a highly strict clinical verification system."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1024
    )
    
    return result.label