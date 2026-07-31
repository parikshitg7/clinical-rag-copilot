"""create_documents_and_chunks_tables

Revision ID: c239fe919c6f
Revises: 
Create Date: 2026-07-31 10:11:45.080633

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c239fe919c6f'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable the pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # 2. Create documents table (using JSONB for flexibility)
    op.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            sections JSONB NOT NULL,
            metadata JSONB NOT NULL
        );
    """)
    
    # 3. Create chunks table with a placeholder vector column
    op.execute("""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY,
            parent_doc_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
            section TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            embedding vector(768)
        );
    """)

def downgrade() -> None:
    op.execute('DROP TABLE chunks;')
    op.execute('DROP TABLE documents;')
    op.execute('DROP EXTENSION IF EXISTS vector;')
