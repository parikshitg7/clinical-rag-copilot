import json
import logging
from pathlib import Path
from src.schema import Claim, Label
from src.verifier import verify_claim

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardcoded test dataset for claim verification
VERIFIER_DATASET = [
    {
        "id": "1",
        "claim_text": "Metformin significantly lowers HbA1c levels in type 2 diabetes patients.",
        "source_text": "Clinical trials consistently demonstrate that Metformin is highly effective as a first-line treatment, significantly reducing HbA1c and fasting plasma glucose in patients with type 2 diabetes.",
        "gold_label": "SUPPORTED"
    },
    {
        "id": "2",
        "claim_text": "Metformin increases the risk of severe hypertension.",
        "source_text": "Metformin is generally well-tolerated. It has a neutral effect on blood pressure, and no studies have linked it to an increased risk of hypertension.",
        "gold_label": "UNSUPPORTED"
    },
    {
        "id": "3",
        "claim_text": "SGLT2 inhibitors are completely safe to use during the third trimester of pregnancy.",
        "source_text": "SGLT2 inhibitors are a class of prescription medicines that are FDA-approved for use with diet and exercise to lower blood sugar in adults with type 2 diabetes.",
        "gold_label": "NOT_ENOUGH_INFO"
    },
    {
        "id": "4",
        "claim_text": "SGLT2 inhibitors reduce hospitalizations for heart failure.",
        "source_text": "Recent cardiovascular outcome trials have shown that SGLT2 inhibitors significantly reduce the risk of hospitalization for heart failure across a broad range of patients.",
        "gold_label": "SUPPORTED"
    },
    {
        "id": "5",
        "claim_text": "Taking ibuprofen daily cures asthma.",
        "source_text": "Ibuprofen is a nonsteroidal anti-inflammatory drug (NSAID) used to relieve pain, reduce fever, and reduce inflammation. It is not indicated for the treatment of asthma and can exacerbate symptoms in some asthmatic patients.",
        "gold_label": "UNSUPPORTED"
    }
]

def is_hallucination(label) -> bool:
    """Helper to treat UNSUPPORTED and NOT_ENOUGH_INFO as the 'Positive' (Hallucination) class."""
    # Handles both Enum objects and raw strings, and forces uppercase for safe comparison
    label_str = label.value if hasattr(label, 'value') else str(label).replace("Label.", "")
    return label_str.upper() in ["UNSUPPORTED", "NOT_ENOUGH_INFO"]

def run_verifier_evaluation(output_file: str = "reports/verifier_metrics.json"):
    logger.info("Starting Verifier Evaluation (Phase 8.3)...")
    
    true_positives = 0  # Caught a bad claim
    false_positives = 0 # Flagged a good claim as bad (paranoid)
    true_negatives = 0  # Let a good claim pass
    false_negatives = 0 # Let a bad claim pass (dangerous)

    for item in VERIFIER_DATASET:
        logger.info(f"Evaluating Claim {item['id']}...")
        
        # Create the Pydantic Claim object
        claim = Claim(
            text=item["claim_text"],
            source_id="mock_source",
            source_chunk_id=1
        )
        
        # Call your actual Groq verifier
        try:
            predicted_label = verify_claim(claim, item["source_text"])
            
            is_gold_bad = item["gold_label"] in ["UNSUPPORTED", "NOT_ENOUGH_INFO"]
            is_pred_bad = is_hallucination(predicted_label)
            
            if is_gold_bad and is_pred_bad:
                true_positives += 1
            elif not is_gold_bad and is_pred_bad:
                false_positives += 1
            elif not is_gold_bad and not is_pred_bad:
                true_negatives += 1
            elif is_gold_bad and not is_pred_bad:
                false_negatives += 1
                
        except Exception as e:
            logger.error(f"Error evaluating claim {item['id']}: {e}")

    # Prevent division by zero
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    report = {
        "dataset_size": len(VERIFIER_DATASET),
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4)
        },
        "confusion_matrix": {
            "true_positives_caught_hallucinations": true_positives,
            "false_positives_paranoid_flags": false_positives,
            "true_negatives_good_claims_passed": true_negatives,
            "false_negatives_missed_hallucinations": false_negatives
        }
    }

    # Save to file
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info(f"\n--- VERIFIER EVALUATION COMPLETE ---")
    logger.info(f"Precision: {report['metrics']['precision']}")
    logger.info(f"Recall:    {report['metrics']['recall']}")
    logger.info(f"F1 Score:  {report['metrics']['f1_score']}")
    
    return report

if __name__ == "__main__":
    run_verifier_evaluation()