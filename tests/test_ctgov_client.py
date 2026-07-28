import responses
from src.ctgov_client import fetch_ctgov_trials, CTGOV_API_URL

@responses.activate
def test_fetch_ctgov_trials_success(tmp_path):
    """Tests that the CT.gov client successfully fetches and saves data using mocked HTTP calls."""
    
    # Mock the API response
    mock_json_data = {
        "studies": [
            {"protocolSection": {"identificationModule": {"nctId": "NCT01234567"}}},
            {"protocolSection": {"identificationModule": {"nctId": "NCT07654321"}}}
        ]
    }
    
    responses.add(
        responses.GET,
        CTGOV_API_URL,
        json=mock_json_data,
        status=200
    )
    
    # Use pytest's tmp_path fixture for a temporary isolated output directory
    output_dir = tmp_path / "data/raw"
    
    # Execute
    record_count = fetch_ctgov_trials("test condition", page_size=2, output_dir=str(output_dir))
    
    # Assertions
    assert record_count == 2
    
    saved_file = output_dir / "ctgov_raw.json"
    assert saved_file.exists()
    
    with open(saved_file, "r") as f:
        import json
        content = json.load(f)
        assert len(content["studies"]) == 2
        assert content["studies"][0]["protocolSection"]["identificationModule"]["nctId"] == "NCT01234567"