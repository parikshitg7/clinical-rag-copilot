import json
import logging
from pathlib import Path
from typing import List, Dict

# Import your precise pipeline components
from src.eval.loader import load_eval_questions
from src.embedding_service import generate_query_embedding, rerank_chunks
from src.repository import search_similar_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(gold_ids: List[str], retrieved_ids: List[str], k: int) -> Dict[str, float]:
    """Calculates Precision@k, Recall@k, and MRR for a single query."""
    if not gold_ids:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
        
    top_k_ids = retrieved_ids[:k]
    
    # Hits in top k (unique parent documents)
    hits = [doc_id for doc_id in top_k_ids if doc_id in gold_ids]
    unique_hits = set(hits)
    
    precision = len(hits) / k if k > 0 else 0.0
    recall = len(unique_hits) / len(gold_ids)
    
    mrr = 0.0
    for rank, doc_id in enumerate(top_k_ids, start=1):
        if doc_id in gold_ids:
            mrr = 1.0 / rank
            break
            
    return {"precision": precision, "recall": recall, "mrr": mrr}

def run_retrieval_evaluation(initial_k: int = 15, eval_k: int = 5, output_file: str = "reports/retrieval_metrics.json"):
    """Runs evaluation over the dataset and compares baseline vs reranked results."""
    logger.info("Starting Retrieval Evaluation (Phase 8.2)...")
    questions = load_eval_questions()
    
    if not questions:
        logger.warning("No evaluation questions found.")
        return {}

    baseline_totals = {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
    reranked_totals = {"precision": 0.0, "recall": 0.0, "mrr": 0.0}
    num_q = len(questions)

    for q in questions:
        logger.info(f"Evaluating Q: '{q.question[:40]}...'")
        
        # 1. Retrieve baseline chunks (using the Query Encoder!)
        query_vector = generate_query_embedding(q.question)
        baseline_chunks = search_similar_chunks(query_vector, limit=initial_k)
        baseline_ids = [chunk["parent_doc_id"] for chunk in baseline_chunks]

        # 2. Rerank chunks (using the Cross-Encoder)
        reranked_chunks = rerank_chunks(q.question, baseline_chunks, top_n=initial_k)
        reranked_ids = [chunk["parent_doc_id"] for chunk in reranked_chunks]

        # 3. Compute metrics strictly on top 'eval_k' (e.g., Top 5)
        base_metrics = compute_metrics(q.gold_source_ids, baseline_ids, eval_k)
        rerank_metrics = compute_metrics(q.gold_source_ids, reranked_ids, eval_k)

        for key in baseline_totals:
            baseline_totals[key] += base_metrics[key]
            reranked_totals[key] += rerank_metrics[key]

    # Average the results
    final_report = {
        "config": {
            "num_questions": num_q,
            "initial_retrieval_k": initial_k,
            "evaluation_k": eval_k
        },
        "baseline_metrics": {k: round(v / num_q, 4) for k, v in baseline_totals.items()},
        "reranked_metrics": {k: round(v / num_q, 4) for k, v in reranked_totals.items()}
    }

    # Save to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)

    logger.info(f"\n--- EVALUATION COMPLETE ---")
    logger.info(f"Baseline Top-{eval_k}: {final_report['baseline_metrics']}")
    logger.info(f"Reranked Top-{eval_k}: {final_report['reranked_metrics']}")
    logger.info(f"Report saved to: {output_file}")
    
    return final_report

if __name__ == "__main__":
    # Ensure fixtures path is correctly resolved when run directly
    run_retrieval_evaluation()