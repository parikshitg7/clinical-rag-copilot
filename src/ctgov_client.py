import os
import requests
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CTGOV_API_URL = "https://clinicaltrials.gov/api/v2/studies"

def fetch_ctgov_trials(query: str, page_size: int = 5, output_dir: str = "data/raw"):
    """
    Fetches raw clinical trial data from ClinicalTrials.gov and saves it to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    params = {
        "query.cond": query, # Querying by condition
        "pageSize": page_size,
        "format": "json"
    }
    
    logger.info(f"Searching ClinicalTrials.gov for condition: {query}")
    
    try:
        response = requests.get(CTGOV_API_URL, params=params)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch from ClinicalTrials.gov: {e}")
        return 0

    data = response.json()
    studies = data.get("studies", [])
    
    if not studies:
        logger.warning("No clinical trials found.")
        return 0
        
    logger.info(f"Found {len(studies)} trials. Saving raw data...")
    
    # Save raw data to disk
    output_path = os.path.join(output_dir, "ctgov_raw.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    logger.info(f"Saved raw JSON to {output_path}")
    return len(studies)

if __name__ == "__main__":
    # Hardcoded query for the script run
    fetch_ctgov_trials(query="hypertension", page_size=5)