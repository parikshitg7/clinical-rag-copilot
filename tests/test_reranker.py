from src.embedding_service import rerank_chunks

def test_reranker_ordering():
    """
    Ensures the MedCPT Cross-Encoder correctly identifies and ranks 
    the most relevant chunk highest for a given synthetic query.
    """
    synthetic_query = "What is the recommended dosage of Metformin for type 2 diabetes?"
    
    # We provide 3 synthetic chunks. 
    # Chunk 2 is the obvious correct answer. Chunk 1 is irrelevant. Chunk 3 is partially relevant.
    synthetic_chunks = [
        {"id": 1, "text": "The patient complained of a headache and was prescribed Ibuprofen."},
        {"id": 2, "text": "Metformin is typically started at 500 mg once or twice daily with meals to treat type 2 diabetes."},
        {"id": 3, "text": "Metformin is an oral antidiabetic medication discovered in the 1920s."}
    ]
    
    # We ask the reranker to sort them and give us the top 3
    reranked = rerank_chunks(synthetic_query, synthetic_chunks, top_n=3)
    
    # Definition of Done: Check that the obvious correct answer (ID 2) was sorted to the very top
    assert reranked[0]["id"] == 2
    
    # Ensure the lowest ranked item is the irrelevant one (ID 1)
    assert reranked[2]["id"] == 1
    
    # Verify the scores were successfully attached
    assert "rerank_score" in reranked[0]