import os
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def fetch_pubmed_abstracts(query: str, retmax: int = 5, output_dir: str = "data/raw"):
    """
    Fetches raw article data from PubMed and saves it to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: esearch to get PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax
    }
    
    logger.info(f"Searching PubMed for: {query}")
    search_resp = requests.get(PUBMED_ESEARCH_URL, params=search_params)
    search_resp.raise_for_status()
    
    id_list = search_resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        logger.warning("No results found.")
        return 0
        
    logger.info(f"Found {len(id_list)} articles. Fetching raw data...")
    
    # Step 2: efetch to get the actual abstracts (raw XML)
    fetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "xml"
    }
    
    fetch_resp = requests.get(PUBMED_EFETCH_URL, params=fetch_params)
    fetch_resp.raise_for_status()
    
    # Save raw data to disk
    output_path = os.path.join(output_dir, "pubmed_raw.xml")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(fetch_resp.text)
        
    logger.info(f"Saved raw XML to {output_path}")
    return len(id_list)

if __name__ == "__main__":
    # Hardcoded query for the script run
    fetch_pubmed_abstracts(query="hypertension treatment", retmax=5)