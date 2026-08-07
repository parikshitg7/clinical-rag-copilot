import json
import logging
from pathlib import Path
from typing import List

from src.eval.loader import load_eval_questions
from src.embedding_service import generate_query_embedding
from src.repository import search_similar_chunks
from src.schema import Chunk, ClinicalAnswer
from src.generator import generate_clinical_answer
from src.graph import clinical_graph
from src.verifier import verify_claim
from src.eval.verifier_metrics import is_hallucination

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_answer_claims(answer: ClinicalAnswer, chunks: List[Chunk]) -> tuple[int, int]:
    """Runs the verifier over every claim in an answer to count unsupported claims."""
    chunk_lookup = {c.chunk_index: c.text for c in chunks}
    total_claims = len(answer.claims)
    unsupported_count = 0
    
    for claim in answer.claims:
        chunk_text = chunk_lookup.get(claim.source_chunk_id, "")
        if not chunk_text:
            unsupported_count += 1
            continue
            
        try:
            label = verify_claim(claim, chunk_text)
            if is_hallucination(label):
                unsupported_count += 1
        except Exception as e:
            logger.error(f"Error evaluating claim: {e}")
            unsupported_count += 1 # Penalize errors as unsupported
            
    return total_claims, unsupported_count

def run_e2e_evaluation(output_file: str = "reports/e2e_metrics.json"):
    logger.info("Starting End-to-End System Evaluation (Phase 8.4)...")
    questions = load_eval_questions()
    
    baseline_total_claims = 0
    baseline_unsupported = 0
    
    graph_total_claims = 0
    graph_unsupported = 0

    for idx, q in enumerate(questions, 1):
        logger.info(f"Processing Q{idx}/{len(questions)}: '{q.question[:40]}...'")
        
        # 1. Retrieve chunks from DB
        query_vector = generate_query_embedding(q.question)
        raw_chunks = search_similar_chunks(query_vector, limit=5)
        
        # Convert DB dicts to Pydantic Chunks safely
        chunks = [
            Chunk(
                parent_doc_id=c["parent_doc_id"], 
                chunk_index=c["chunk_index"], 
                text=c["text"], 
                section=c.get("section", "Unknown")
            ) 
            for c in raw_chunks
        ]
        
        if not chunks:
            logger.warning("No chunks found. Skipping.")
            continue
            
        # 2. Condition A: Baseline RAG (No Verification)
        logger.info("  -> Running Condition A: Baseline Generator")
        baseline_answer = generate_clinical_answer(q.question, chunks)
        b_total, b_unsup = evaluate_answer_claims(baseline_answer, chunks)
        baseline_total_claims += b_total
        baseline_unsupported += b_unsup
        
        # 3. Condition B: Verified LangGraph RAG
        logger.info("  -> Running Condition B: LangGraph Agent")
        graph_state = clinical_graph.invoke({
            "question": q.question, 
            "chunks": chunks, 
            "retries": 0, 
            "messages": []
        })
        graph_answer = graph_state["answer"]
        g_total, g_unsup = evaluate_answer_claims(graph_answer, chunks)
        graph_total_claims += g_total
        graph_unsupported += g_unsup

    # 4. Calculate Final Metrics
    baseline_rate = (baseline_unsupported / baseline_total_claims) if baseline_total_claims > 0 else 0.0
    graph_rate = (graph_unsupported / graph_total_claims) if graph_total_claims > 0 else 0.0

    report = {
        "dataset_size": len(questions),
        "baseline_rag": {
            "total_claims_made": baseline_total_claims,
            "unsupported_claims": baseline_unsupported,
            "hallucination_rate": round(baseline_rate, 4)
        },
        "verified_langgraph": {
            "total_claims_made": graph_total_claims,
            "unsupported_claims": graph_unsupported,
            "hallucination_rate": round(graph_rate, 4)
        },
        "improvement": {
            "absolute_reduction": round(baseline_rate - graph_rate, 4)
        }
    }

    # Save to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("\n=== FINAL SYSTEM HEADLINE METRICS ===")
    logger.info(f"Baseline Hallucination Rate: {baseline_rate * 100:.1f}%")
    logger.info(f"LangGraph Hallucination Rate: {graph_rate * 100:.1f}%")
    logger.info("=====================================")
    
    return report

if __name__ == "__main__":
    run_e2e_evaluation()