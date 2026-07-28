from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class Document(BaseModel):
    """Normalized representation of a clinical document."""
    id: str
    source: str
    title: Optional[str] = "Untitled"
    sections: Dict[str, str] = Field(default_factory=dict)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """A retrievable unit of text derived from a Document."""
    parent_doc_id: str
    section: str
    chunk_index: int
    text: str