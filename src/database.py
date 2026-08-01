import psycopg2
import os

# Fetch the URL from the environment (used in Docker).
# If it doesn't exist, fall back to localhost for local terminal testing.
DB_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://postgres:password@localhost:5432/clinical_rag"
)

def get_connection():
    """Returns a connection to the Postgres database."""
    conn = psycopg2.connect(DB_URL)
    return conn