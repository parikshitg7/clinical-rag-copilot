import os
import responses
from src.pubmed_client import fetch_pubmed_abstracts, PUBMED_ESEARCH_URL, PUBMED_EFETCH_URL

@responses.activate
def test_fetch_pubmed_abstracts_success(tmp_path):
    """Tests that the client successfully fetches and saves data using mocked HTTP calls."""
    
    # 1. Mock the esearch response
    mock_search_response = {
        "esearchresult": {
            "idlist": ["12345", "67890"]
        }
    }
    responses.add(
        responses.GET,
        PUBMED_ESEARCH_URL,
        json=mock_search_response,
        status=200
    )
    
    # 2. Mock the efetch response
    mock_xml_data = "<PubmedArticleSet><Article><ID>12345</ID></Article></PubmedArticleSet>"
    responses.add(
        responses.GET,
        PUBMED_EFETCH_URL,
        body=mock_xml_data,
        status=200
    )
    
    # Use pytest's tmp_path fixture for a temporary isolated output directory
    output_dir = tmp_path / "data/raw"
    
    # Execute
    record_count = fetch_pubmed_abstracts("test query", retmax=2, output_dir=str(output_dir))
    
    # Assertions
    assert record_count == 2
    
    saved_file = output_dir / "pubmed_raw.xml"
    assert saved_file.exists()
    
    with open(saved_file, "r") as f:
        content = f.read()
        assert "PubmedArticleSet" in content