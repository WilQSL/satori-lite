"""
Shared test fixtures for performance and load tests.
"""
import pytest
import requests


@pytest.fixture(scope="session")
def server_url():
    """Base URL for the test server."""
    return "http://localhost:8000"


@pytest.fixture(scope="session")
def server_available(server_url):
    """
    Check if server is available. Skip tests if not running.

    For performance tests, server MUST be running.
    """
    try:
        response = requests.get(f"{server_url}/health", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    pytest.skip(
        f"Server not available at {server_url}. "
        "Performance tests require a running server. "
        "Start server with: export DATABASE_URL='sqlite:///:memory:' && "
        "uvicorn src.main:app --host 0.0.0.0 --port 8000"
    )


@pytest.fixture
def challenge_token(server_url):
    """Get a fresh challenge token for tests."""
    response = requests.get(f"{server_url}/api/v1/auth/challenge")
    return response.json()["challenge"]
