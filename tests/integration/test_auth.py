"""
Integration tests for authentication flow.

Tests the challenge-response authentication mechanism with the actual server.
"""
import pytest
import requests


@pytest.mark.integration
@pytest.mark.auth
def test_get_challenge_token(server_url, server_available):
    """Test getting a challenge token from the server."""
    response = requests.get(f"{server_url}/api/v1/auth/challenge")

    assert response.status_code == 200
    data = response.json()
    assert "challenge" in data
    assert isinstance(data["challenge"], str)
    assert len(data["challenge"]) > 0


@pytest.mark.integration
@pytest.mark.auth
def test_challenge_token_is_unique(server_url, server_available):
    """Test that each challenge request returns a unique token."""
    response1 = requests.get(f"{server_url}/api/v1/auth/challenge")
    response2 = requests.get(f"{server_url}/api/v1/auth/challenge")

    assert response1.status_code == 200
    assert response2.status_code == 200

    token1 = response1.json()["challenge"]
    token2 = response2.json()["challenge"]

    # Tokens should be different
    assert token1 != token2


@pytest.mark.integration
@pytest.mark.auth
def test_challenge_token_format(server_url, server_available):
    """Test challenge token is in UUID format."""
    response = requests.get(f"{server_url}/api/v1/auth/challenge")

    assert response.status_code == 200
    challenge = response.json()["challenge"]

    # UUID v4 format: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx
    parts = challenge.split('-')
    assert len(parts) == 5, "UUID should have 5 parts separated by hyphens"
    assert len(parts[0]) == 8
    assert len(parts[1]) == 4
    assert len(parts[2]) == 4
    assert len(parts[3]) == 4
    assert len(parts[4]) == 12


@pytest.mark.integration
@pytest.mark.auth
def test_client_can_get_challenge(client_instance, server_available):
    """Test SatoriServerClient can get challenge token."""
    challenge = client_instance._getChallenge()

    assert challenge is not None
    assert isinstance(challenge, str)
    assert len(challenge) > 0


@pytest.mark.integration
@pytest.mark.auth
def test_authenticated_call_with_mock_wallet(server_url, server_available, test_wallet):
    """Test authenticated call with test wallet (will fail auth but tests flow)."""
    # Get challenge
    challenge_response = requests.get(f"{server_url}/api/v1/auth/challenge")
    challenge = challenge_response.json()["challenge"]

    # Create auth headers
    headers = test_wallet.authPayload(asDict=True, challenge=challenge)

    # Try authenticated endpoint (will fail because mock wallet signature invalid)
    response = requests.get(
        f"{server_url}/api/v1/balance/get",
        headers=headers
    )

    # Should get 401 because signature is invalid (expected for mock wallet)
    # This tests that the authentication flow is being checked
    assert response.status_code == 401


@pytest.mark.integration
@pytest.mark.auth
def test_missing_auth_headers_rejected(server_url, server_available):
    """Test that requests without auth headers are rejected."""
    # Try to access authenticated endpoint without headers
    response = requests.get(f"{server_url}/api/v1/balance/get")

    # Should get 422 or 401 (missing required headers)
    assert response.status_code in [401, 422]


@pytest.mark.integration
@pytest.mark.auth
def test_invalid_pubkey_format_rejected(server_url, server_available, challenge_token):
    """Test that invalid public key format is rejected."""
    headers = {
        "wallet-pubkey": "invalid_pubkey",  # Invalid format
        "message": challenge_token,
        "signature": "fake_signature"
    }

    response = requests.post(
        f"{server_url}/api/v1/prediction/post",
        headers=headers,
        json={"value": "100.0"}
    )

    # Should reject invalid pubkey format
    assert response.status_code in [400, 401]
