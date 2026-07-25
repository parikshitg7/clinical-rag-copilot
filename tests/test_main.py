from src.main import health_check

def test_health_check():
    """Proves that pytest is wired up correctly."""
    assert health_check() is True