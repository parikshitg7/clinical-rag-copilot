from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class Document(BaseModel):
    """Normalized representation of a clinical document."""
    id: str
    source: str
    title: Optional[str] = "Untitled"
    sections: Dict[str, str] = Field(default_factory=dict)
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)