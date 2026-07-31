import sqlite3
import json
from src.database import get_connection
from src.schema import Document, Chunk

def save_document(doc: Document) -> None:
    """Saves a single Document to the SQLite database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO documents (id, source, title, sections, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (
        doc.id,
        doc.source,
        doc.title,
        json.dumps(doc.sections),
        json.dumps(doc.metadata)
    ))
    
    conn.commit()
    conn.close()

def save_chunks(chunks: list[Chunk]) -> None:
    """Saves a list of Chunks to the SQLite database."""
    if not chunks:
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    for chunk in chunks:
        cursor.execute("""
            INSERT INTO chunks (parent_doc_id, section, chunk_index, text)
            VALUES (?, ?, ?, ?)
        """, (
            chunk.parent_doc_id,
            chunk.section,
            chunk.chunk_index,
            chunk.text
        ))
        
    conn.commit()
    conn.close()

def get_chunks_by_doc_id(doc_id: str) -> list[Chunk]:
    """Retrieves all Chunks for a given Document ID."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT parent_doc_id, section, chunk_index, text 
        FROM chunks 
        WHERE parent_doc_id = ?
        ORDER BY chunk_index ASC
    """, (doc_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        Chunk(
            parent_doc_id=row["parent_doc_id"],
            section=row["section"],
            chunk_index=row["chunk_index"],
            text=row["text"]
        ) for row in rows
    ]