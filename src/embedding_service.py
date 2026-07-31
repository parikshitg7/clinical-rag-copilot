from sentence_transformers import SentenceTransformer
from typing import List

# We are using the exact MedCPT Article Encoder dictated by the Project Spec
MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

# Load the model (this will download the model weights the first time it runs)
model = SentenceTransformer(MODEL_NAME)

def generate_embedding(text: str) -> List[float]:
    """
    Converts a string of text into a mathematical vector (embedding) 
    using the MedCPT Article Encoder.
    """
    # Generate the embedding and convert the numpy array to a standard Python list of floats
    embedding = model.encode(text)
    return embedding.tolist()