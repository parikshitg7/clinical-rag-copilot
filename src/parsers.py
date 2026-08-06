import xml.etree.ElementTree as ET
from typing import Dict, Any
from src.schema import Document

def parse_pubmed_xml_record(xml_string: str) -> Document:
    """Parses a single raw PubMed XML article string into a Document."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return Document(id="unknown", source="pubmed", text="")
    
    pmid_elem = root.find(".//PMID")
    doc_id = pmid_elem.text if pmid_elem is not None else "unknown"
    
    title_elem = root.find(".//ArticleTitle")
    title = title_elem.text if title_elem is not None else "Untitled"
    
    # Extract abstract sections
    sections = {}
    abstract_texts = root.findall(".//AbstractText")
    full_text_parts = []
    
    for text_node in abstract_texts:
        label = text_node.attrib.get("Label", "UNLABELED")
        content = text_node.text or ""
        sections[label] = content
        full_text_parts.append(f"{label}: {content}" if label != "UNLABELED" else content)
    
    full_text = "\n".join(full_text_parts).strip()
    
    return Document(
        id=doc_id,
        source="pubmed",
        title=title,
        sections=sections,
        text=full_text
    )