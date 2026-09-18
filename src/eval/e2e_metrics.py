import json
import logging
import time
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

CHECKPOINT_FILE = "reports/checkpoint_e2e.json"
FINAL_REPORT_FILE = "reports/e2e_metrics.json"

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
            time.sleep(2)
            label = verify_claim(claim, chunk_text)
            if is_hallucination(label):
                unsupported_count += 1
        except Exception as e:
            logger.error(f"Error evaluating claim: {e}")
            unsupported_count += 1
            
    return total_claims, unsupported_count

def load_checkpoint() -> dict:
    """Loads existing progress if present."""
    path = Path(CHECKPOINT_FILE)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"evaluations": []}

def save_checkpoint(data: dict):
    """Saves evaluation results after each question."""
    path = Path(CHECKPOINT_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def run_e2e_evaluation():
    logger.info("Starting End-to-End System Evaluation (Phase 8.4) with gpt-oss-120b...")
    questions = load_eval_questions()
    
    checkpoint_data = load_checkpoint()
    evaluated_ids = {item["id"] for item in checkpoint_data["evaluations"]}
    logger.info(f"Loaded {len(evaluated_ids)} previously evaluated questions from checkpoint.")

    for idx, q in enumerate(questions, 1):
        if q.id in evaluated_ids:
            logger.info(f"Skipping Q{idx}/{len(questions)} ({q.id}) - Already Evaluated.")
            continue

        logger.info(f"Processing Q{idx}/{len(questions)}: '{q.question[:40]}...'")
        
        try:
            # 1. Retrieve chunks from DB
            query_vector = generate_query_embedding(q.question)
            raw_chunks = search_similar_chunks(query_vector, limit=5)
            
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
                
            # 2. Condition A: Baseline RAG
            logger.info("  -> Running Condition A: Baseline Generator")
            time.sleep(2)
            baseline_answer = generate_clinical_answer(q.question, chunks)
            b_total, b_unsup = evaluate_answer_claims(baseline_answer, chunks)
            
            # 3. Condition B: Verified LangGraph RAG
            logger.info("  -> Running Condition B: LangGraph Agent")
            time.sleep(3)
            graph_state = clinical_graph.invoke({
                "question": q.question, 
                "chunks": chunks, 
                "retries": 0, 
                "messages": []
            })
            graph_answer = graph_state["answer"]
            g_total, g_unsup = evaluate_answer_claims(graph_answer, chunks)

            # Record single item result
            q_result = {
                "id": q.id,
                "question": q.question,
                "baseline": {"total_claims": b_total, "unsupported": b_unsup},
                "langgraph": {"total_claims": g_total, "unsupported": g_unsup}
            }
            checkpoint_data["evaluations"].append(q_result)
            save_checkpoint(checkpoint_data)
            
            logger.info("  -> Pausing 15 between questions for rate-limit protection...")
            time.sleep(15)

        except Exception as e:
            logger.error(f"Execution stopped on Q{idx} ({q.id}) due to error or rate limit: {e}")
            logger.info("Progress saved! You can run this script again to resume.")
            break

    # Calculate metrics over all currently completed questions
    evals = checkpoint_data["evaluations"]
    if not evals:
        logger.warning("No evaluations completed yet.")
        return

    b_claims = sum(item["baseline"]["total_claims"] for item in evals)
    b_unsup = sum(item["baseline"]["unsupported"] for item in evals)
    
    g_claims = sum(item["langgraph"]["total_claims"] for item in evals)
    g_unsup = sum(item["langgraph"]["unsupported"] for item in evals)

    baseline_rate = (b_unsup / b_claims) if b_claims > 0 else 0.0
    graph_rate = (g_unsup / g_claims) if g_claims > 0 else 0.0

    report = {
        "questions_evaluated_so_far": len(evals),
        "total_target_questions": len(questions),
        "baseline_rag": {
            "total_claims_made": b_claims,
            "unsupported_claims": b_unsup,
            "hallucination_rate": round(baseline_rate, 4)
        },
        "verified_langgraph": {
            "total_claims_made": g_claims,
            "unsupported_claims": g_unsup,
            "hallucination_rate": round(graph_rate, 4)
        },
        "improvement": {
            "absolute_reduction": round(baseline_rate - graph_rate, 4)
        }
    }

    out_path = Path(FINAL_REPORT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("\n=== CURRENT SYSTEM HEADLINE METRICS ===")
    logger.info(f"Questions Evaluated: {len(evals)} / {len(questions)}")
    logger.info(f"Baseline Hallucination Rate: {baseline_rate * 100:.1f}%")
    logger.info(f"LangGraph Hallucination Rate: {graph_rate * 100:.1f}%")
    logger.info("=======================================")

if __name__ == "__main__":
    run_e2e_evaluation()