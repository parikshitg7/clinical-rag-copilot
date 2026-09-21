import logging
from src.repository import get_unembedded_chunks, update_chunk_embedding
from src.embedding_service import generate_embedding

logging.basicConfig(level=logging.INFO)

def run_batch_job():
    chunks = get_unembedded_chunks()
    total = len(chunks)
    logging.info(f"Found {total} unembedded chunks.")

    for i, chunk in enumerate(chunks, 1):
        chunk_id = chunk["id"]
        text = chunk["text"]
        
        logging.info(f"Embedding chunk {i} of {total} (ID: {chunk_id})...")
        
        try:
            vector = generate_embedding(text)
            update_chunk_embedding(chunk_id, vector)
        except Exception as e:
            logging.error(f"Failed to embed chunk {chunk_id}: {e}")
            # Retry with aggressive truncation if an unexpected length error occurs
            try:
                logging.info(f"Retrying chunk {chunk_id} with aggressive truncation...")
                truncated_text = text[:1000]
                vector = generate_embedding(truncated_text)
                update_chunk_embedding(chunk_id, vector)
            except Exception as retry_e:
                logging.error(f"Skipping chunk {chunk_id} after retry failed: {retry_e}")
                continue

if __name__ == "__main__":
    run_batch_job()