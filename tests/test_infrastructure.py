"""
Test infrastructure smoke tests.

Verifies that the test setup is working correctly.
"""
import pytest


def test_pytest_works():
    """Verify pytest is working."""
    assert True


def test_fixtures_available(test_wallet, test_server_url, client_instance):
    """Verify that common fixtures are available."""
    assert test_wallet is not None
    assert test_server_url is not None
    assert client_instance is not None


def test_wallet_fixture_structure(test_wallet):
    """Verify test wallet has required attributes."""
    assert hasattr(test_wallet, 'publicKey')
    assert hasattr(test_wallet, 'address')
    assert hasattr(test_wallet, 'sign')
    assert hasattr(test_wallet, 'verify')
    assert hasattr(test_wallet, 'authPayload')

    # Verify publicKey format (66 chars, hex, starts with 02 or 03)
    assert len(test_wallet.publicKey) == 66
    assert test_wallet.publicKey.startswith(('02', '03'))


def test_client_instance_fixture(client_instance):
    """Verify client instance has required attributes."""
    assert hasattr(client_instance, 'wallet')
    assert hasattr(client_instance, 'url')
    assert hasattr(client_instance, 'publish')
    assert hasattr(client_instance, '_getChallenge')
    assert hasattr(client_instance, '_makeAuthenticatedCall')


@pytest.mark.unit
def test_unit_marker():
    """Verify unit test marker works."""
    assert True


@pytest.mark.integration
def test_integration_marker():
    """Verify integration test marker works."""
    assert True
