"""
Integration tests for health and basic endpoints.

These tests make actual HTTP requests to the server.
"""
import pytest
import requests


@pytest.mark.integration
def test_server_root_endpoint(server_url, server_available):
    """Test GET / returns API information."""
    response = requests.get(f"{server_url}/")

    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "Satori"
    assert "version" in data


@pytest.mark.integration
def test_server_health_endpoint(server_url, server_available):
    """Test GET /health returns healthy status."""
    response = requests.get(f"{server_url}/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data


@pytest.mark.integration
def test_server_health_timestamp_format(server_url, server_available):
    """Test /health returns ISO format timestamp."""
    response = requests.get(f"{server_url}/health")

    assert response.status_code == 200
    data = response.json()

    timestamp = data.get("timestamp")
    assert timestamp is not None

    # Should be ISO format: YYYY-MM-DDTHH:MM:SS.ffffff
    assert "T" in timestamp
    assert len(timestamp) > 10


@pytest.mark.integration
def test_server_responds_quickly(server_url, server_available):
    """Test server responds within reasonable time."""
    import time

    start = time.time()
    response = requests.get(f"{server_url}/health")
    elapsed = time.time() - start

    assert response.status_code == 200
    # Should respond within 1 second
    assert elapsed < 1.0
