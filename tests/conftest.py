"""
Shared pytest fixtures for satori-lite tests.

Provides common fixtures for unit and integration tests.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path

# Add satori-lite lib-lite to Python path
SATORI_LITE_PATH = Path(__file__).parent.parent / "lib-lite"
if str(SATORI_LITE_PATH) not in sys.path:
    sys.path.insert(0, str(SATORI_LITE_PATH))


@pytest.fixture
def test_server_url():
    """
    URL for the test server.

    Returns the server URL from environment or defaults to localhost:8000.
    Integration tests should set this to a running server instance.
    """
    return os.environ.get("SATORI_SERVER_URL", "http://localhost:8000")


@pytest.fixture
def test_wallet():
    """
    Create a test wallet for authentication.

    For integration tests, this creates a real wallet that can sign messages.
    For unit tests, this can be mocked.

    Returns:
        Mock wallet with required attributes for testing:
        - address: Test wallet address
        - publicKey: Test public key (66 char compressed hex)
        - privkey: Test private key
        - sign(): Method to sign messages
        - verify(): Method to verify signatures
        - authPayload(): Method to create auth payload
    """
    # Create a mock wallet with realistic test data
    wallet = MagicMock()

    # Use a valid compressed Evrmore public key format (66 chars, starts with 02 or 03)
    wallet.publicKey = "026a2a2bee6d2d1b4db0fb60c20a60a3bcfee53ef911d3e20ce0ebc079006558c2"
    wallet.address = "ETestAddress123456789ABCDEFGHIJK"
    wallet.privkey = "test_private_key_for_testing"

    # Mock sign method to return a valid signature
    def mock_sign(message: str) -> bytes:
        return f"signature_of_{message}".encode()

    wallet.sign = Mock(side_effect=mock_sign)

    # Mock verify method
    wallet.verify = Mock(return_value=True)

    # Mock authPayload method to match real wallet behavior
    def mock_auth_payload(asDict=False, challenge=None):
        payload = {
            "wallet-pubkey": wallet.publicKey,
            "message": challenge or "test_message",
            "signature": "test_signature"
        }
        if asDict:
            return payload
        return payload

    wallet.authPayload = Mock(side_effect=mock_auth_payload)

    return wallet


@pytest.fixture
def client_instance(test_wallet, test_server_url):
    """
    Create a SatoriServerClient instance for testing.

    Args:
        test_wallet: Test wallet fixture
        test_server_url: Server URL fixture

    Returns:
        SatoriServerClient configured with test wallet and server URL
    """
    from satorilib.server.server import SatoriServerClient

    client = SatoriServerClient(
        wallet=test_wallet,
        url=test_server_url
    )

    return client


@pytest.fixture
def mock_requests():
    """
    Mock for requests library to use in unit tests.

    Returns:
        Mock requests module with get, post, delete methods
    """
    requests_mock = MagicMock()

    # Mock response object
    response_mock = MagicMock()
    response_mock.status_code = 200
    response_mock.text = '{"status": "ok"}'
    response_mock.json.return_value = {"status": "ok"}

    # Mock HTTP methods
    requests_mock.get.return_value = response_mock
    requests_mock.post.return_value = response_mock
    requests_mock.delete.return_value = response_mock
    requests_mock.put.return_value = response_mock

    return requests_mock


@pytest.fixture
def mock_response():
    """
    Create a mock HTTP response for unit tests.

    Returns:
        Mock response object with common attributes
    """
    response = MagicMock()
    response.status_code = 200
    response.text = '{"message": "success"}'
    response.json.return_value = {"message": "success"}

    def raise_for_status():
        if response.status_code >= 400:
            from requests.exceptions import HTTPError
            raise HTTPError(f"HTTP {response.status_code}")

    response.raise_for_status = Mock(side_effect=raise_for_status)

    return response


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests with mocked dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring running server"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time to run"
    )
