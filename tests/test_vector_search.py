from src.embedding_service import generate_query_embedding
from src.repository import search_similar_chunks

def test_vector_search_retrieves_chunks():
    """
    Ensures that a user query can be embedded and used to retrieve 
    stored chunks from the Postgres database.
    """
    user_question = "What is the primary use of this medication?"
    
    # 1. Generate the query vector
    query_vector = generate_query_embedding(user_question)
    
    # 2. Assert dimensionality (MedCPT Query is also 768 dimensions)
    assert len(query_vector) == 768
    
    # 3. Perform the database search (looking for top 2 results)
    results = search_similar_chunks(query_vector, limit=2)
    
    # 4. Verify we get results back (we expect the 2 chunks from Phase 3.1)
    assert isinstance(results, list)
    assert len(results) > 0
    
    # 5. Verify the structure of the returned chunks
    first_result = results[0]
    assert "text" in first_result
    assert "similarity" in first_result
    
    # Similarity should be a float between -1 and 1
    assert isinstance(first_result["similarity"], float)