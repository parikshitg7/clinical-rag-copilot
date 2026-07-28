from src.parsers import parse_pubmed_xml_record, parse_ctgov_json_record
from src.schema import Document

def test_parse_pubmed_success():
    """Tests normal PubMed XML parsing."""
    xml = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345</PMID>
            <Article>
                <ArticleTitle>Test Title</ArticleTitle>
                <Abstract>
                    <AbstractText Label="BACKGROUND">A test background.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>
    """
    doc = parse_pubmed_xml_record(xml)
    assert doc.id == "12345"
    assert doc.title == "Test Title"
    assert doc.source == "pubmed"
    assert "BACKGROUND" in doc.sections

def test_parse_pubmed_missing_fields():
    """Tests PubMed parser resilience against missing titles and abstracts."""
    xml = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID>999</PMID>
        </MedlineCitation>
    </PubmedArticle>
    """
    doc = parse_pubmed_xml_record(xml)
    assert doc.id == "999"
    assert doc.title == "Untitled"  # Default fallback
    assert doc.text == ""           # Empty abstract handled gracefully

def test_parse_ctgov_success():
    """Tests normal CT.gov JSON parsing."""
    data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT123",
                "briefTitle": "CT Title"
            },
            "descriptionModule": {
                "briefSummary": "A brief summary."
            }
        }
    }
    doc = parse_ctgov_json_record(data)
    assert doc.id == "NCT123"
    assert doc.title == "CT Title"
    assert doc.source == "clinicaltrials.gov"
    assert doc.text == "A brief summary."

def test_parse_ctgov_missing_fields():
    """Tests CT.gov parser resilience against missing descriptions."""
    data = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT456"
            }
            # descriptionModule entirely missing
        }
    }
    doc = parse_ctgov_json_record(data)
    assert doc.id == "NCT456"
    assert doc.title == "Untitled"
    assert doc.text == ""
    