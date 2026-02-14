"""Tests for Satori Nostr data models."""
import json
import pytest
import time
from satori_nostr.models import (
    DatastreamMetadata,
    DatastreamObservation,
    SubscriptionAnnouncement,
    PaymentNotification,
    InboundObservation,
    InboundPayment,
    SatoriNostrConfig,
    compute_stream_topic_tag,
)


class TestDatastreamMetadata:
    """Tests for DatastreamMetadata model."""

    def test_create_metadata(self):
        """Test creating datastream metadata."""
        metadata = DatastreamMetadata(
            stream_id="btc-price-usd",
            neuron_pubkey="abc123",
            name="Bitcoin Price (USD)",
            description="Real-time BTC/USD price from Coinbase",
            encrypted=True,
            price_per_obs=10,
            created_at=1234567890,
            tags=["bitcoin", "price", "usd"],
        )

        assert metadata.stream_id == "btc-price-usd"
        assert metadata.neuron_pubkey == "abc123"
        assert metadata.name == "Bitcoin Price (USD)"
        assert metadata.encrypted is True
        assert metadata.price_per_obs == 10
        assert len(metadata.tags) == 3

    def test_serialize_to_dict(self):
        """Test serializing metadata to dictionary."""
        metadata = DatastreamMetadata(
            stream_id="btc-price-usd",
            neuron_pubkey="abc123",
            name="Bitcoin Price",
            description="BTC/USD",
            encrypted=False,
            price_per_obs=0,
            created_at=1234567890,
            tags=["bitcoin"],
        )

        data = metadata.to_dict()
        assert isinstance(data, dict)
        assert data["stream_id"] == "btc-price-usd"
        assert data["encrypted"] is False
        assert data["price_per_obs"] == 0

    def test_deserialize_from_dict(self):
        """Test deserializing metadata from dictionary."""
        data = {
            "stream_id": "eth-price",
            "neuron_pubkey": "def456",
            "name": "ETH Price",
            "description": "Ethereum price",
            "encrypted": True,
            "price_per_obs": 5,
            "created_at": 1234567890,
            "tags": ["ethereum", "price"],
        }

        metadata = DatastreamMetadata.from_dict(data)
        assert metadata.stream_id == "eth-price"
        assert metadata.encrypted is True
        assert len(metadata.tags) == 2

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = DatastreamMetadata(
            stream_id="test-stream",
            neuron_pubkey="pubkey123",
            name="Test Stream",
            description="Testing",
            encrypted=False,
            price_per_obs=0,
            created_at=1234567890,
            tags=["test"],
        )

        json_str = original.to_json()
        restored = DatastreamMetadata.from_json(json_str)

        assert restored.stream_id == original.stream_id
        assert restored.neuron_pubkey == original.neuron_pubkey
        assert restored.encrypted == original.encrypted
        assert restored.tags == original.tags


class TestDatastreamObservation:
    """Tests for DatastreamObservation model."""

    def test_create_observation(self):
        """Test creating an observation."""
        obs = DatastreamObservation(
            stream_id="btc-price-usd",
            timestamp=int(time.time()),
            value={"price": 45000.50, "volume": 123.45},
            seq_num=1,
        )

        assert obs.stream_id == "btc-price-usd"
        assert obs.seq_num == 1
        assert obs.value["price"] == 45000.50

    def test_observation_with_various_value_types(self):
        """Test observations with different value types."""
        # Dict value
        obs1 = DatastreamObservation(
            stream_id="test", timestamp=123, value={"a": 1}, seq_num=1
        )
        assert isinstance(obs1.value, dict)

        # Number value
        obs2 = DatastreamObservation(
            stream_id="test", timestamp=123, value=42.5, seq_num=2
        )
        assert isinstance(obs2.value, float)

        # String value
        obs3 = DatastreamObservation(
            stream_id="test", timestamp=123, value="hello", seq_num=3
        )
        assert isinstance(obs3.value, str)

        # List value
        obs4 = DatastreamObservation(
            stream_id="test", timestamp=123, value=[1, 2, 3], seq_num=4
        )
        assert isinstance(obs4.value, list)

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = DatastreamObservation(
            stream_id="test-stream",
            timestamp=1234567890,
            value={"price": 100, "volume": 50},
            seq_num=42,
        )

        json_str = original.to_json()
        restored = DatastreamObservation.from_json(json_str)

        assert restored.stream_id == original.stream_id
        assert restored.timestamp == original.timestamp
        assert restored.value == original.value
        assert restored.seq_num == original.seq_num


class TestSubscriptionAnnouncement:
    """Tests for SubscriptionAnnouncement model."""

    def test_create_subscription(self):
        """Test creating a subscription announcement."""
        sub = SubscriptionAnnouncement(
            subscriber_pubkey="sub123",
            stream_id="btc-price-usd",
            provider_pubkey="provider456",
            timestamp=int(time.time()),
            payment_channel="lightning:channel123",
        )

        assert sub.subscriber_pubkey == "sub123"
        assert sub.stream_id == "btc-price-usd"
        assert sub.provider_pubkey == "provider456"
        assert sub.payment_channel == "lightning:channel123"

    def test_subscription_without_payment_channel(self):
        """Test subscription without payment channel (free stream)."""
        sub = SubscriptionAnnouncement(
            subscriber_pubkey="sub123",
            stream_id="free-stream",
            provider_pubkey="provider456",
            timestamp=1234567890,
        )

        assert sub.payment_channel is None

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = SubscriptionAnnouncement(
            subscriber_pubkey="sub",
            stream_id="stream",
            provider_pubkey="provider",
            timestamp=1234567890,
            payment_channel="lightning:xyz",
        )

        json_str = original.to_json()
        restored = SubscriptionAnnouncement.from_json(json_str)

        assert restored.subscriber_pubkey == original.subscriber_pubkey
        assert restored.stream_id == original.stream_id
        assert restored.payment_channel == original.payment_channel


class TestPaymentNotification:
    """Tests for PaymentNotification model."""

    def test_create_payment(self):
        """Test creating a payment notification."""
        payment = PaymentNotification(
            from_pubkey="subscriber123",
            to_pubkey="provider456",
            stream_id="btc-price-usd",
            seq_num=42,
            amount_sats=10,
            timestamp=int(time.time()),
            tx_id="lightning:tx789",
        )

        assert payment.from_pubkey == "subscriber123"
        assert payment.to_pubkey == "provider456"
        assert payment.stream_id == "btc-price-usd"
        assert payment.seq_num == 42
        assert payment.amount_sats == 10
        assert payment.tx_id == "lightning:tx789"

    def test_payment_without_tx_id(self):
        """Test payment without transaction ID."""
        payment = PaymentNotification(
            from_pubkey="sub",
            to_pubkey="provider",
            stream_id="stream",
            seq_num=1,
            amount_sats=5,
            timestamp=123,
        )

        assert payment.tx_id is None

    def test_json_round_trip(self):
        """Test JSON serialization round-trip."""
        original = PaymentNotification(
            from_pubkey="sub",
            to_pubkey="provider",
            stream_id="stream",
            seq_num=10,
            amount_sats=100,
            timestamp=1234567890,
            tx_id="tx123",
        )

        json_str = original.to_json()
        restored = PaymentNotification.from_json(json_str)

        assert restored.from_pubkey == original.from_pubkey
        assert restored.seq_num == original.seq_num
        assert restored.amount_sats == original.amount_sats
        assert restored.tx_id == original.tx_id


class TestInboundModels:
    """Tests for inbound message wrapper models."""

    def test_inbound_observation(self):
        """Test InboundObservation wrapper."""
        obs = DatastreamObservation(
            stream_id="test",
            timestamp=123,
            value={"data": 42},
            seq_num=1,
        )

        inbound = InboundObservation(
            stream_id="test",
            provider_pubkey="provider123",
            observation=obs,
            event_id="event456",
        )

        assert inbound.stream_id == "test"
        assert inbound.provider_pubkey == "provider123"
        assert inbound.observation.value["data"] == 42
        assert inbound.event_id == "event456"
        assert inbound.raw_event is None

    def test_inbound_payment(self):
        """Test InboundPayment wrapper."""
        payment = PaymentNotification(
            from_pubkey="sub",
            to_pubkey="provider",
            stream_id="stream",
            seq_num=5,
            amount_sats=10,
            timestamp=123,
        )

        inbound = InboundPayment(
            payment=payment,
            event_id="event789",
            raw_event={"kind": 30103},
        )

        assert inbound.payment.seq_num == 5
        assert inbound.event_id == "event789"
        assert inbound.raw_event["kind"] == 30103


class TestSatoriNostrConfig:
    """Tests for SatoriNostrConfig."""

    def test_create_config_minimal(self):
        """Test creating config with minimal required fields."""
        config = SatoriNostrConfig(
            keys="nsec1...",
            relay_urls=["wss://relay.damus.io"],
        )

        assert config.keys == "nsec1..."
        assert len(config.relay_urls) == 1
        assert config.active_relay_timeout_ms == 8000  # default
        assert config.dedupe_db_path is None  # default

    def test_create_config_full(self):
        """Test creating config with all fields."""
        config = SatoriNostrConfig(
            keys="nsec1...",
            relay_urls=["wss://relay1.io", "wss://relay2.io"],
            active_relay_timeout_ms=5000,
            dedupe_db_path="/tmp/dedupe.db",
        )

        assert len(config.relay_urls) == 2
        assert config.active_relay_timeout_ms == 5000
        assert config.dedupe_db_path == "/tmp/dedupe.db"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_compute_stream_topic_tag(self):
        """Test computing stream topic tags."""
        tag1 = compute_stream_topic_tag("btc-price-usd")
        assert tag1 == "satori:stream:btc-price-usd"

        tag2 = compute_stream_topic_tag("eth-price")
        assert tag2 == "satori:stream:eth-price"

        # Different streams have different tags
        assert tag1 != tag2

    def test_topic_tag_format(self):
        """Test topic tag has correct format."""
        tag = compute_stream_topic_tag("my-stream")
        assert tag.startswith("satori:stream:")
        assert "my-stream" in tag
