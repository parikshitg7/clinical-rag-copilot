import logging
from src.repository import get_unembedded_chunks, update_chunk_embedding
from src.embedding_service import generate_embedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_batch_job():
    """Finds all unembedded chunks and generates vectors for them."""
    chunks = get_unembedded_chunks()
    logger.info(f"Found {len(chunks)} chunks needing embeddings.")

    if not chunks:
        logger.info("No chunks to embed. Exiting.")
        return

    for index, chunk in enumerate(chunks):
        logger.info(f"Embedding chunk {index + 1} of {len(chunks)} (ID: {chunk['id']})...")
        vector = generate_embedding(chunk['text'])
        update_chunk_embedding(chunk['id'], vector)

    logger.info("Batch embedding complete. All chunks are vectorized.")

if __name__ == "__main__":
    run_batch_job()