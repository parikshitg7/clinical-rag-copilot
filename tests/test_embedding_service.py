from src.embedding_service import generate_embedding

def test_embedding_dimensionality_and_type():
    """
    Ensures the embedding service returns a list of floats 
    with exactly 768 dimensions (the spec for MedCPT Article Encoder).
    """
    sample_medical_text = "Metformin is an oral antidiabetic medication used to treat type 2 diabetes."
    
    # Generate the vector
    vector = generate_embedding(sample_medical_text)
    
    # Verify the output is a list
    assert isinstance(vector, list)
    
    # Verify it has exactly 768 dimensions
    assert len(vector) == 768
    
    # Verify the contents are floats
    assert isinstance(vector[0], float)