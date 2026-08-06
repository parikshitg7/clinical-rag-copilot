import pytest
from src.parsers import parse_pubmed_xml_record
from src.schema import Document

def test_parse_valid_pubmed_xml():
    # Setup: A well-formed PubMed XML snippet
    valid_xml = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID Version="1">12345678</PMID>
            <Article>
                <ArticleTitle>Test Medical Treatment of Hypertension</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">Hypertension is common.</AbstractText>
                    <AbstractText Label="RESULTS">Treatment reduced risk.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """
    
    # Execution
    doc = parse_pubmed_xml_record(valid_xml)
    
    # Assertions
    assert isinstance(doc, Document)
    assert doc.id == "12345678"
    assert doc.source == "pubmed"
    assert doc.title == "Test Medical Treatment of Hypertension"
    assert "BACKGROUND" in doc.sections
    assert doc.sections["BACKGROUND"] == "Hypertension is common."
    assert "RESULTS" in doc.sections
    assert doc.sections["RESULTS"] == "Treatment reduced risk."
    assert len(doc.text) > 0

def test_parse_malformed_xml_edge_case():
    # Edge Case: Malformed or broken XML string
    malformed_xml = "<PubmedArticle><UnclosedTag>Oops"
    
    # Execution
    doc = parse_pubmed_xml_record(malformed_xml)
    
    # Assertions: Should handle gracefully and return a fallback Document rather than crashing
    assert isinstance(doc, Document)
    assert doc.id == "unknown"
    assert doc.source == "pubmed"
    assert doc.text == ""