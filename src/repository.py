import psycopg2
from psycopg2.extras import RealDictCursor
from src.database import get_connection
from src.schema import Document, Chunk
import json

def save_document(doc: Document):
    """Saves a Document to the Postgres database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO documents (id, source, title, sections, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            source = EXCLUDED.source,
            title = EXCLUDED.title,
            sections = EXCLUDED.sections,
            metadata = EXCLUDED.metadata;
    """, (doc.id, doc.source, doc.title, json.dumps(doc.sections), json.dumps(doc.metadata)))
    conn.commit()
    cursor.close()
    conn.close()

def save_chunks(chunks: list[Chunk]):
    """Saves a list of Chunks to the Postgres database."""
    conn = get_connection()
    cursor = conn.cursor()
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (parent_doc_id, section, chunk_index, text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT DO NOTHING;
        """, (chunk.parent_doc_id, chunk.section, chunk.chunk_index, chunk.text))
    conn.commit()
    cursor.close()
    conn.close()

def get_chunks_by_doc_id(doc_id: str) -> list[Chunk]:
    """Retrieves all Chunks for a specific Document ID."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM chunks WHERE parent_doc_id = %s ORDER BY chunk_index;", (doc_id,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return [Chunk(**row) for row in rows]

def get_unembedded_chunks() -> list[dict]:
    """Fetches all chunks that do not yet have a vector embedding."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    # Assumes your Alembic migration named the vector column 'embedding'
    cursor.execute("SELECT id, text FROM chunks WHERE embedding IS NULL;")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def update_chunk_embedding(chunk_id: int, embedding: list[float]):
    """Updates a specific chunk with its generated vector embedding."""
    conn = get_connection()
    cursor = conn.cursor()
    # Cast the list of floats to a string format that pgvector accepts
    cursor.execute("""
        UPDATE chunks SET embedding = %s WHERE id = %s;
    """, (str(embedding), chunk_id))
    conn.commit()
    cursor.close()
    conn.close()

def search_similar_chunks(query_embedding: list[float], limit: int = 5) -> list[dict]:
    """
    Searches the database for chunks closest to the query embedding 
    using pgvector's cosine distance operator (<=>).
    """
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # We calculate 1 - distance to get a 'similarity score' (higher is better)
    # We order by the closest distance ascending.
    cursor.execute("""
        SELECT id, parent_doc_id, section, chunk_index, text, 
               1 - (embedding <=> %s::vector) AS similarity 
        FROM chunks 
        ORDER BY embedding <=> %s::vector 
        LIMIT %s;
    """, (str(query_embedding), str(query_embedding), limit))
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows
