# src/graph.py
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from src.schema import Chunk, ClinicalAnswer, Claim, Label
from src.generator import generate_clinical_answer, client, SYSTEM_PROMPT
from src.verifier import verify_claim

# 1. Define the State that gets passed between nodes
class AgentState(TypedDict):
    question: str
    chunks: List[Chunk]
    answer: ClinicalAnswer
    retries: int
    messages: List[str]  # Used to store feedback from the verifier

# 2. Node: Generate the initial answer
def generate_node(state: AgentState):
    # This node only runs on the very first pass
    answer = generate_clinical_answer(state["question"], state["chunks"])
    return {"answer": answer, "retries": 0, "messages": []}

# 3. Node: Verify all claims
def verify_node(state: AgentState):
    answer = state["answer"]
    chunks = state["chunks"]
    
    # Create a fast lookup dictionary for chunk text by its index
    chunk_lookup = {c.chunk_index: c.text for c in chunks}
    
    supported_claims = []
    unsupported_claims = []
    
    for claim in answer.claims:
        chunk_text = chunk_lookup.get(claim.source_chunk_id, "")
        if not chunk_text:
            unsupported_claims.append((claim, "Source chunk not found in retrieved context."))
            continue
            
        label = verify_claim(claim, chunk_text)
        if label == Label.SUPPORTED:
            supported_claims.append(claim)
        else:
            unsupported_claims.append((claim, label.value))
            
    # Formulate feedback for the LLM if there are failures
    feedback = ""
    if unsupported_claims:
        feedback = "The following claims were UNSUPPORTED by their cited chunks:\n"
        for c, reason in unsupported_claims:
            feedback += f"- Claim: '{c.text}' (Cited Chunk ID: {c.source_chunk_id}). Reason: {reason}\n"
        feedback += "Please rewrite the answer. Fix these claims by finding better support in the context, or completely drop them if they cannot be supported."
        
    # We update the answer to only contain supported claims. 
    # If the graph exits here, the user only sees verified truths.
    answer.claims = supported_claims 
    if not unsupported_claims:
        answer.verified = True
        
    return {"answer": answer, "messages": [feedback] if feedback else []}

# 4. Node: Regenerate the answer using feedback
def regenerate_node(state: AgentState):
    question = state["question"]
    chunks = state["chunks"]
    feedback = state["messages"][-1]
    
    context_text = "\n\n".join(
        [f"[Chunk ID: {c.chunk_index} | Source: {c.parent_doc_id}]\n{c.text}" for c in chunks]
    )
    
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}\n\nFeedback from Verifier:\n{feedback}"
    
    # Direct LLM call using your existing patched client and system prompt
    new_answer = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_model=ClinicalAnswer,
        temperature=0.0,
    )
    
    return {"answer": new_answer, "retries": state["retries"] + 1}

# 5. Conditional Edge: Decide where to go after verification
def route_verification(state: AgentState):
    # If there is no feedback, everything passed! Exit the loop.
    if not state.get("messages") or not state["messages"][-1]:
        return "finalize"
        
    # If we hit the retry cap (e.g., 2 retries), stop trying to fix it and exit.
    if state["retries"] >= 2:
        return "finalize"
        
    # Otherwise, send it back for regeneration.
    return "regenerate"

# --- Build and Compile the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("generate", generate_node)
workflow.add_node("verify", verify_node)
workflow.add_node("regenerate", regenerate_node)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "verify")
workflow.add_conditional_edges(
    "verify",
    route_verification,
    {
        "finalize": END,
        "regenerate": "regenerate"
    }
)
workflow.add_edge("regenerate", "verify")

clinical_graph = workflow.compile()