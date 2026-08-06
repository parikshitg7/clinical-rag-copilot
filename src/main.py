import os
import logging
import xml.etree.ElementTree as ET
from dotenv import load_dotenv

# Import pipeline components matching your exact architecture
from src.pubmed_client import fetch_pubmed_abstracts
from src.parsers import parse_pubmed_xml_record
from src.chunker import chunk_document
from src.repository import save_document, save_chunks
from src.batch_embed import run_batch_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_ingestion_pipeline(query: str, max_results: int = 5):
    """Runs the end-to-end ingestion pipeline using PubMed data."""
    logger.info(f"--- Starting PubMed Ingestion Pipeline for query: '{query}' ---")
    
    # 1. Fetch raw XML data from PubMed
    num_fetched = fetch_pubmed_abstracts(query=query, retmax=max_results)
    if num_fetched == 0:
        logger.warning("No articles fetched from PubMed. Exiting pipeline.")
        return

    xml_path = "data/raw/pubmed_raw.xml"
    if not os.path.exists(xml_path):
        logger.error(f"Expected raw data file at {xml_path} but it was not found.")
        return

    # 2. Parse bulk XML file
    logger.info("Parsing bulk XML file...")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # PubMed wraps articles in <PubmedArticle> tags
    articles = root.findall(".//PubmedArticle")
    logger.info(f"Found {len(articles)} articles in XML. Processing...")

    for article in articles:
        xml_string = ET.tostring(article, encoding="unicode")
        
        # 3. Parse into Document schema
        doc = parse_pubmed_xml_record(xml_string)
        if doc.id == "unknown" or not doc.text.strip():
            logger.warning("Skipping malformed or empty document.")
            continue
            
        logger.info(f"Processing PMID: {doc.id} - {doc.title[:30]}...")
        
        # 4. Save Document to Postgres database
        save_document(doc)
        
        # 5. Chunk Document
        chunks = chunk_document(doc)
        
        # 6. Save Chunks to Postgres database
        save_chunks(chunks)
        
    # 7. Generate Vector Embeddings for new chunks
    logger.info("Triggering vector embeddings batch job...")
    run_batch_job()
    
    logger.info("--- PubMed Ingestion Pipeline Complete! ---")

if __name__ == "__main__":
    load_dotenv()
    run_ingestion_pipeline(query="hypertension treatment", max_results=5)