import xml.etree.ElementTree as ET
from typing import Dict, Any
from src.schema import Document

def parse_pubmed_xml_record(xml_string: str) -> Document:
    """Parses a single raw PubMed XML article string into a Document."""
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        # Handle malformed XML edge case
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

def parse_ctgov_json_record(trial_dict: Dict[str, Any]) -> Document:
    """Parses a single raw ClinicalTrials.gov JSON dictionary into a Document."""
    protocol = trial_dict.get("protocolSection", {})
    ident_module = protocol.get("identificationModule", {})
    desc_module = protocol.get("descriptionModule", {})
    
    nct_id = ident_module.get("nctId", "unknown")
    title = ident_module.get("officialTitle", ident_module.get("briefTitle", "Untitled"))
    
    brief_summary = desc_module.get("briefSummary", "")
    detailed_desc = desc_module.get("detailedDescription", "")
    
    sections = {}
    if brief_summary:
        sections["Brief Summary"] = brief_summary
    if detailed_desc:
        sections["Detailed Description"] = detailed_desc
        
    full_text = f"{brief_summary}\n{detailed_desc}".strip()
    
    return Document(
        id=nct_id,
        source="clinicaltrials.gov",
        title=title,
        sections=sections,
        text=full_text
    )