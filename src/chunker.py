from typing import List
from src.schema import Document, Chunk

def chunk_document(doc: Document, max_words: int = 250) -> List[Chunk]:
    """
    Splits a Document into Chunk objects by section.
    Approximates token limits by using word counts (~250 words is ~330 tokens).
    """
    chunks = []
    global_chunk_index = 0

    # If the document has no specific sections, treat the full text as one "General" section
    sections_to_chunk = doc.sections if doc.sections else {"General": doc.text}

    for section_name, section_text in sections_to_chunk.items():
        if not section_text.strip():
            continue
            
        words = section_text.split()
        
        # Handle short documents/sections that fit in a single chunk
        if len(words) <= max_words:
            chunks.append(
                Chunk(
                    parent_doc_id=doc.id,
                    section=section_name,
                    chunk_index=global_chunk_index,
                    text=" ".join(words)
                )
            )
            global_chunk_index += 1
            continue

        # Handle long documents/sections by splitting them up
        current_chunk_words = []
        
        for word in words:
            current_chunk_words.append(word)
            if len(current_chunk_words) >= max_words:
                chunks.append(
                    Chunk(
                        parent_doc_id=doc.id,
                        section=section_name,
                        chunk_index=global_chunk_index,
                        text=" ".join(current_chunk_words)
                    )
                )
                global_chunk_index += 1
                current_chunk_words = []
                
        # Catch any remaining words at the end of the section
        if current_chunk_words:
            chunks.append(
                Chunk(
                    parent_doc_id=doc.id,
                    section=section_name,
                    chunk_index=global_chunk_index,
                    text=" ".join(current_chunk_words)
                )
            )
            global_chunk_index += 1

    return chunks