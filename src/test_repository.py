import pytest
import os
from src.database import init_db, get_connection, DB_PATH
from src.schema import Document, Chunk
from src.repository import save_document, save_chunks, get_chunks_by_doc_id

@pytest.fixture(autouse=True)
def setup_test_db():
    """Ensure we are working with a clean test database."""
    # Use an in-memory DB or a separate test file. 
    # For simplicity, we'll re-init and clear the local file for tests.
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    yield

def test_repository_round_trip():
    # 1. Setup mock data
    doc = Document(
        id="PMID:12345",
        source="PubMed",
        title="Test Medical Trial",
        sections={"abstract": "This is a test abstract."},
        text="This is a test abstract.",
        metadata={"year": 2023}
    )
    
    chunks = [
        Chunk(parent_doc_id="PMID:12345", section="abstract", chunk_index=0, text="This is a test"),
        Chunk(parent_doc_id="PMID:12345", section="abstract", chunk_index=1, text="abstract.")
    ]
    
    # 2. Save Document and Chunks to SQLite
    save_document(doc)
    save_chunks(chunks)
    
    # 3. Retrieve chunks back from SQLite
    retrieved_chunks = get_chunks_by_doc_id("PMID:12345")
    
    # 4. Assert round-trip success
    assert len(retrieved_chunks) == 2
    assert retrieved_chunks[0].text == "This is a test"
    assert retrieved_chunks[1].text == "abstract."
    assert retrieved_chunks[0].parent_doc_id == "PMID:12345"