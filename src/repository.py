import json
from src.database import get_connection
from src.schema import Document, Chunk
from psycopg2.extras import RealDictCursor

def save_document(doc: Document) -> None:
    """Saves a single Document to the Postgres database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Postgres uses %s for variables and ON CONFLICT for upserts
    cursor.execute("""
        INSERT INTO documents (id, source, title, sections, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            source = EXCLUDED.source,
            title = EXCLUDED.title,
            sections = EXCLUDED.sections,
            metadata = EXCLUDED.metadata;
    """, (
        doc.id,
        doc.source,
        doc.title,
        json.dumps(doc.sections),
        json.dumps(doc.metadata)
    ))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_chunks(chunks: list[Chunk]) -> None:
    """Saves a list of Chunks to the Postgres database."""
    if not chunks:
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (parent_doc_id, section, chunk_index, text)
            VALUES (%s, %s, %s, %s)
        """, (
            chunk.parent_doc_id,
            chunk.section,
            chunk.chunk_index,
            chunk.text
        ))
        
    conn.commit()
    cursor.close()
    conn.close()

def get_chunks_by_doc_id(doc_id: str) -> list[Chunk]:
    """Retrieves all Chunks for a given Document ID."""
    conn = get_connection()
    # RealDictCursor allows us to access columns by name (e.g. row["text"])
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT parent_doc_id, section, chunk_index, text 
        FROM chunks 
        WHERE parent_doc_id = %s
        ORDER BY chunk_index ASC
    """, (doc_id,))
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [
        Chunk(
            parent_doc_id=row["parent_doc_id"],
            section=row["section"],
            chunk_index=row["chunk_index"],
            text=row["text"]
        ) for row in rows
    ]