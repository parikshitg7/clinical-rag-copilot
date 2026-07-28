from src.schema import Document
from src.chunker import chunk_document

def test_chunk_short_document():
    """Tests that a short document produces exactly one chunk."""
    doc = Document(
        id="DOC1",
        source="test",
        sections={"Background": "This is a short background."},
        text="This is a short background."
    )
    chunks = chunk_document(doc, max_words=10)
    
    assert len(chunks) == 1
    assert chunks[0].parent_doc_id == "DOC1"
    assert chunks[0].section == "Background"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "This is a short background."

def test_chunk_long_document_and_data_loss():
    """Tests splitting a long document and ensures no data is lost."""
    # Create a string of 25 words
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    
    doc = Document(
        id="DOC2",
        source="test",
        sections={"Methods": text},
        text=text
    )
    
    # Chunk with a max of 10 words per chunk
    chunks = chunk_document(doc, max_words=10)
    
    # 25 words / 10 words per chunk = 3 chunks (10, 10, 5)
    assert len(chunks) == 3
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[2].chunk_index == 2
    
    assert len(chunks[0].text.split()) == 10
    assert len(chunks[2].text.split()) == 5
    
    # Data loss check: re-combining the chunks must equal the original words
    reconstructed_text = " ".join([c.text for c in chunks])
    assert reconstructed_text == text

def test_chunk_section_boundaries():
    """Tests that chunks respect section boundaries."""
    doc = Document(
        id="DOC3",
        source="test",
        sections={
            "Background": "Short background.",
            "Results": "Short results."
        },
        text="Short background.\nShort results."
    )
    
    chunks = chunk_document(doc, max_words=10)
    
    assert len(chunks) == 2
    assert chunks[0].section == "Background"
    assert chunks[0].text == "Short background."
    
    assert chunks[1].section == "Results"
    assert chunks[1].text == "Short results."