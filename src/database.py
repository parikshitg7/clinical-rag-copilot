import os
import psycopg2
from dotenv import load_dotenv

# Load local .env file if present
load_dotenv()

def get_connection():
    """Returns a connection to the Postgres database."""
    db_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:password@localhost:5432/clinical_rag"
    )
    return psycopg2.connect(db_url)