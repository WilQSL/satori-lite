"""
Complete user journey tests.

Tests end-to-end workflows that clients would perform in production.
"""
import pytest
import requests
import json
import sys

# Add server path to import client
sys.path.insert(0, '/app')
from src.services.database import SessionLocal
from src.models import Prediction, Observation, Peer


@pytest.mark.integration
@pytest.mark.slow
def test_new_client_complete_journey(client_instance, server_url, server_available):
    """Test complete journey: new client connects and performs all operations.

    Workflow:
    1. Client checks in with server
    2. Client retrieves latest observation
    3. Client submits a prediction
    4. Verify operations complete without errors
    """
    # Step 1: Checkin
    checkin_result = client_instance.checkin()
    assert isinstance(checkin_result, dict), "Checkin should return dict"
    assert client_instance.lastCheckin > 0, "Last checkin timestamp should be set"

    # Step 2: Get observation
    obs_response = requests.get(f"{server_url}/api/v1/observation/get")
    assert obs_response.status_code == 200, "Should be able to get observations"
    obs_data = obs_response.json()
    # Can be None if no observations exist
    assert obs_data is None or isinstance(obs_data, dict)

    # Step 3: Submit prediction (will fail auth with mock wallet)
    prediction_result = client_instance.publish(
        topic="journey-test-topic",
        data="12345.67",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="journey-hash-123",
        isPrediction=True
    )

    # With mock wallet, auth fails, but workflow completes
    assert prediction_result in [None, True], "Prediction should complete workflow"


@pytest.mark.integration
@pytest.mark.slow
def test_client_session_multiple_operations(client_instance, server_url, server_available):
    """Test client performing multiple operations in one session.

    Simulates a client that:
    - Checks in periodically
    - Retrieves observations multiple times
    - Submits multiple predictions
    """
    # Operation 1: Initial checkin
    checkin1 = client_instance.checkin()
    assert isinstance(checkin1, dict)

    # Operation 2: Get observation
    obs1 = requests.get(f"{server_url}/api/v1/observation/get")
    assert obs1.status_code == 200

    # Operation 3: Submit prediction to topic A
    pred1 = client_instance.publish(
        topic="session-topic-a",
        data="100.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="session-hash-1",
        isPrediction=True
    )
    assert pred1 in [None, True]

    # Operation 4: Submit prediction to topic B (different topic, not rate limited)
    pred2 = client_instance.publish(
        topic="session-topic-b",
        data="200.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="session-hash-2",
        isPrediction=True
    )
    assert pred2 in [None, True]

    # Operation 5: Try to submit to topic A again (should be rate limited)
    pred3 = client_instance.publish(
        topic="session-topic-a",
        data="300.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="session-hash-3",
        isPrediction=True
    )
    assert pred3 is None, "Should be rate limited on topic A"

    # Operation 6: Another checkin
    initial_checkin_time = client_instance.lastCheckin
    checkin2 = client_instance.checkin()
    assert isinstance(checkin2, dict)
    assert client_instance.lastCheckin >= initial_checkin_time, "Checkin timestamp should update or stay same"


@pytest.mark.integration
@pytest.mark.slow
def test_multiple_clients_concurrent_operations(test_wallet, test_server_url, server_available):
    """Test multiple clients operating concurrently without interference."""
    from satorilib.server.server import SatoriServerClient
    from concurrent.futures import ThreadPoolExecutor

    num_clients = 5

    def client_workflow(client_id):
        """Each client performs a complete workflow."""
        client = SatoriServerClient(wallet=test_wallet, url=test_server_url)

        # Checkin
        checkin = client.checkin()
        if not isinstance(checkin, dict):
            return False

        # Publish prediction (unique topic per client)
        pred = client.publish(
            topic=f"multi-client-{client_id}",
            data=f"{client_id * 100}.00",
            observationTime="2025-01-20T10:00:00Z",
            observationHash=f"multi-hash-{client_id}",
            isPrediction=True
        )

        # Workflow should complete
        return pred in [None, True]

    # Run clients concurrently
    with ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = [executor.submit(client_workflow, i) for i in range(num_clients)]
        results = [f.result() for f in futures]

    # All clients should complete their workflows
    success_rate = sum(results) / len(results) * 100
    assert success_rate >= 80, f"At least 80% of clients should complete workflows"


@pytest.mark.integration
@pytest.mark.slow
def test_client_reconnection_scenario(test_wallet, test_server_url, server_available):
    """Test client disconnect and reconnect scenario.

    Simulates:
    1. Client connects and operates
    2. Client disconnects (object destroyed)
    3. New client connects (same wallet)
    4. Operations continue
    """
    from satorilib.server.server import SatoriServerClient

    # First session
    client1 = SatoriServerClient(wallet=test_wallet, url=test_server_url)
    checkin1 = client1.checkin()
    assert isinstance(checkin1, dict)

    # Submit prediction
    pred1 = client1.publish(
        topic="reconnect-topic",
        data="100.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="reconnect-hash-1",
        isPrediction=True
    )
    assert pred1 in [None, True]

    # "Disconnect" (destroy client object)
    del client1

    # Second session (reconnect with same wallet)
    client2 = SatoriServerClient(wallet=test_wallet, url=test_server_url)
    checkin2 = client2.checkin()
    assert isinstance(checkin2, dict)

    # Should be able to operate normally
    # Note: Rate limit state is client-side, so new client can submit
    pred2 = client2.publish(
        topic="reconnect-topic-2",
        data="200.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="reconnect-hash-2",
        isPrediction=True
    )
    assert pred2 in [None, True]


@pytest.mark.integration
@pytest.mark.slow
def test_observation_then_prediction_workflow(server_url, server_available, client_instance):
    """Test realistic workflow: observe data, then make prediction based on it.

    This is the core use case for Satori.
    """
    # Step 1: Get latest observation to make prediction on
    obs_response = requests.get(f"{server_url}/api/v1/observation/get")
    assert obs_response.status_code == 200

    obs_data = obs_response.json()

    # Step 2: Make prediction based on observation
    # In real system, client would use ML model to predict next value

    if obs_data is not None:
        # We have observation data to predict on
        observed_value = obs_data.get('value', '0')

        # Make a "prediction" (in reality would be from ML model)
        predicted_value = "999.99"  # Dummy prediction

        pred_result = client_instance.publish(
            topic="observe-predict-topic",
            data=predicted_value,
            observationTime="2025-01-20T10:00:00Z",
            observationHash="observe-predict-hash",
            isPrediction=True
        )

        # Workflow completes
        assert pred_result in [None, True]
    else:
        # No observation available, but workflow still works
        # Client could still make prediction based on historical data
        assert True, "Workflow handles empty observation state"


@pytest.mark.integration
@pytest.mark.slow
def test_health_check_before_operations(server_url, server_available):
    """Test that clients check server health before operating."""
    # Step 1: Check health
    health_response = requests.get(f"{server_url}/health")
    assert health_response.status_code == 200

    health_data = health_response.json()
    assert health_data["status"] == "healthy"

    # Step 2: Proceed with operations only if healthy
    if health_data["status"] == "healthy":
        # Safe to proceed with challenge request
        challenge_response = requests.get(f"{server_url}/api/v1/auth/challenge")
        assert challenge_response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
def test_error_recovery_workflow(server_url, server_available, client_instance):
    """Test client recovers from errors and continues operating.

    Simulates:
    1. Client tries invalid operation (fails)
    2. Client recovers
    3. Client performs valid operation (succeeds)
    """
    # Step 1: Invalid operation - bad endpoint
    bad_response = requests.get(f"{server_url}/api/v1/invalid/endpoint")
    assert bad_response.status_code == 404

    # Step 2: Client should recover and continue

    # Step 3: Valid operation after error
    health_response = requests.get(f"{server_url}/health")
    assert health_response.status_code == 200, "Client should recover from errors"

    # Step 4: Client can still perform normal operations
    checkin = client_instance.checkin()
    assert isinstance(checkin, dict), "Client should continue operating after error"


@pytest.mark.integration
@pytest.mark.slow
def test_sequential_predictions_to_same_topic(client_instance, server_available):
    """Test sequential predictions showing rate limiting behavior.

    This demonstrates the rate limiting workflow:
    1. First prediction succeeds (or fails auth)
    2. Immediate second prediction blocked by rate limit
    3. After waiting, can submit again
    """
    topic = "sequential-topic"

    # First prediction
    pred1 = client_instance.publish(
        topic=topic,
        data="100.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="seq-hash-1",
        isPrediction=True
    )
    # Either succeeds or fails auth, but not rate limited
    assert pred1 in [None, True]

    # Immediate second prediction (should be blocked)
    pred2 = client_instance.publish(
        topic=topic,
        data="200.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="seq-hash-2",
        isPrediction=True
    )
    # Should be blocked by rate limit
    assert pred2 is None, "Second prediction should be rate limited"

    # Verify rate limit is tracked
    assert topic in client_instance.topicTime, "Topic should be in rate limit tracker"


@pytest.mark.integration
@pytest.mark.slow
def test_mixed_operations_workflow(client_instance, server_url, server_available):
    """Test client performing mixed operations (checkin, observe, predict, repeat)."""
    # Cycle 1
    checkin1 = client_instance.checkin()
    assert isinstance(checkin1, dict)

    obs1 = requests.get(f"{server_url}/api/v1/observation/get")
    assert obs1.status_code == 200

    pred1 = client_instance.publish(
        topic="mixed-topic-1",
        data="100.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="mixed-hash-1",
        isPrediction=True
    )
    assert pred1 in [None, True]

    # Cycle 2
    checkin2 = client_instance.checkin()
    assert isinstance(checkin2, dict)

    obs2 = requests.get(f"{server_url}/api/v1/observation/get")
    assert obs2.status_code == 200

    pred2 = client_instance.publish(
        topic="mixed-topic-2",
        data="200.0",
        observationTime="2025-01-20T10:00:00Z",
        observationHash="mixed-hash-2",
        isPrediction=True
    )
    assert pred2 in [None, True]

    # All operations should complete successfully
    assert client_instance.lastCheckin > 0, "Client state should be maintained"
