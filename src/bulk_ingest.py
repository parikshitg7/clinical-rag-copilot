import os
import time
import logging
import xml.etree.ElementTree as ET
from src.schema import Document
from src.pubmed_client import fetch_pubmed_abstracts
from src.chunker import chunk_document
from src.repository import save_document, save_chunks  # Using your exact repository function name!

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The 8 Major Clinical Domains for a highly dense, robust portfolio RAG
DOMAINS = {
    "Cardiology": '"heart failure"[MeSH Terms] OR "hypertension"[MeSH Terms]',
    "Endocrinology": '"diabetes mellitus"[MeSH Terms] OR "thyroid diseases"[MeSH Terms]',
    "Oncology": '"neoplasms"[MeSH Terms] AND "immunotherapy"[MeSH Terms]',
    "Infectious_Disease": '"communicable diseases"[MeSH Terms] OR "anti-bacterial agents"[MeSH Terms]',
    "Neurology": '"stroke"[MeSH Terms] OR "alzheimer disease"[MeSH Terms]',
    "Pharmacology": '"pharmacokinetics"[MeSH Terms] OR "drug interactions"[MeSH Terms]',
    "Pediatrics": '"pediatrics"[MeSH Terms] OR "childhood diseases"[Title/Abstract]',
    "General_Medicine": '"primary health care"[MeSH Terms] OR "fever"[MeSH Terms] OR "inflammation"[MeSH Terms]'
}

def parse_pubmed_xml(xml_path: str) -> list[Document]:
    """Parses PubMed XML into Pydantic Document objects."""
    docs = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for article in root.findall('.//PubmedArticle'):
            pmid = article.findtext('.//PMID')
            title = article.findtext('.//ArticleTitle')
            abstract_nodes = article.findall('.//AbstractText')
            abstract_text = "\n".join([node.text for node in abstract_nodes if node.text])
            
            if pmid and abstract_text:
                docs.append(Document(
                    id=f"PMID_{pmid}",
                    text=f"{title}\n\n{abstract_text}",
                    title=title or "Untitled",
                    source="PubMed",
                    sections={"Abstract": abstract_text},
                    metadata={"pmid": pmid}
                ))
    except Exception as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
    return docs

def run_mass_ingestion():
    total_chunks_saved = 0
    
    for domain, query in DOMAINS.items():
        logger.info(f"\n=== Starting Ingestion for Domain: {domain} ===")
        output_dir = f"data/raw/{domain}"
        
        # 1. Fetch 150 articles using your existing client
        fetched_count = fetch_pubmed_abstracts(query=query, retmax=180, output_dir=output_dir)
        
        if fetched_count > 0:
            xml_path = os.path.join(output_dir, "pubmed_raw.xml")
            
            # 2. Parse XML into Documents
            docs = parse_pubmed_xml(xml_path)
            logger.info(f"Successfully parsed {len(docs)} documents for {domain}.")
            
            if docs:
                # STEP A: Save Parent Documents one-by-one using your repository function
                for doc in docs:
                    save_document(doc)
                
                # STEP B: Chunk documents
                domain_chunks = []
                for doc in docs:
                    domain_chunks.extend(chunk_document(doc))
                    
                # STEP C: Save Chunks
                if domain_chunks:
                    save_chunks(domain_chunks)
                    total_chunks_saved += len(domain_chunks)
                    logger.info(f"Chunked and saved {len(domain_chunks)} chunks for {domain}.")
        
        # RATE LIMITING PROTECTION: Sleep for 2 seconds to keep the NCBI API happy
        logger.info("Sleeping for 4 seconds to respect PubMed API rate limits...")
        time.sleep(4)

    logger.info(f"\n=== MASS INGESTION COMPLETE ===")
    logger.info(f"Total new chunks added to Postgres: {total_chunks_saved}")

if __name__ == "__main__":
    run_mass_ingestion()