import psycopg2
import os

# Hardcoded for local dev right now; will move to .env later
DB_URL = "postgresql://postgres:password@localhost:5432/clinical_rag"

def get_connection():
    """Returns a connection to the Postgres database."""
    conn = psycopg2.connect(DB_URL)
    return conn