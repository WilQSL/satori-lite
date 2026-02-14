"""Data models for Satori Nostr library.

Models for datastream pub/sub with micropayments over Nostr.
"""
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class DatastreamMetadata:
    """Public metadata about a datastream.

    Published as kind 30100 event (plaintext, discoverable).
    """
    stream_id: str           # Unique identifier (e.g., "btc-price-usd")
    neuron_pubkey: str       # Provider's public key (hex)
    name: str                # Human-readable name
    description: str         # What this stream provides
    encrypted: bool          # Is data encrypted? (True for paid streams)
    price_per_obs: int       # Price in satoshis per observation (0 = free)
    created_at: int          # Unix timestamp
    tags: list[str]          # Searchable tags (e.g., ["bitcoin", "price", "usd"])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatastreamMetadata":
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DatastreamMetadata":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class DatastreamObservation:
    """A single data point in a datastream.

    Published as kind 30101 event (encrypted DM to each paid subscriber).
    """
    stream_id: str           # Which stream this belongs to
    timestamp: int           # Observation time (Unix timestamp)
    value: Any               # The actual data (dict, number, string, list, etc.)
    seq_num: int             # Sequence number for ordering and payment tracking

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stream_id": self.stream_id,
            "timestamp": self.timestamp,
            "value": self.value,
            "seq_num": self.seq_num,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatastreamObservation":
        """Deserialize from dictionary."""
        return cls(
            stream_id=data["stream_id"],
            timestamp=data["timestamp"],
            value=data["value"],
            seq_num=data["seq_num"],
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "DatastreamObservation":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SubscriptionAnnouncement:
    """Public announcement of a datastream subscription.

    Published as kind 30102 event (plaintext, for accountability).
    Lets everyone see who is subscribing to what streams.
    """
    subscriber_pubkey: str   # Who is subscribing (hex)
    stream_id: str           # What they're subscribing to
    provider_pubkey: str     # Who provides the stream (hex)
    timestamp: int           # When subscription started (Unix timestamp)
    payment_channel: str | None = None  # Optional: Lightning channel address

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubscriptionAnnouncement":
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "SubscriptionAnnouncement":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class PaymentNotification:
    """Payment notification from subscriber to provider.

    Published as kind 30103 event (encrypted DM to provider).
    Public metadata visible: stream_id, seq_num, timestamp
    Private: amount (optional), transaction details
    """
    from_pubkey: str         # Subscriber's public key (hex)
    to_pubkey: str           # Provider's public key (hex)
    stream_id: str           # What stream this payment is for
    seq_num: int             # Which observation this payment covers
    amount_sats: int         # Payment amount in satoshis
    timestamp: int           # When payment was sent (Unix timestamp)
    tx_id: str | None = None # Optional: Lightning transaction/payment proof

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PaymentNotification":
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "PaymentNotification":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class InboundObservation:
    """A received observation after decryption and parsing."""
    stream_id: str                      # Which stream it's from
    provider_pubkey: str                # Provider's public key (hex)
    observation: DatastreamObservation  # The actual observation data
    event_id: str                       # Nostr event ID
    raw_event: dict[str, Any] | None = None  # Optional raw Nostr event


@dataclass
class InboundPayment:
    """A received payment notification after decryption."""
    payment: PaymentNotification        # The payment data
    event_id: str                       # Nostr event ID
    raw_event: dict[str, Any] | None = None  # Optional raw Nostr event


@dataclass
class SatoriNostrConfig:
    """Configuration for SatoriNostr client."""

    # Required fields
    keys: str                           # nsec (bech32) or hex private key
    relay_urls: list[str]               # Relay URLs to connect to

    # Optional fields with defaults
    active_relay_timeout_ms: int = 8000  # Failover trigger timeout
    dedupe_db_path: str | None = None    # SQLite path for dedupe (None = in-memory)


# Event kind constants
KIND_DATASTREAM_ANNOUNCE = 30100    # Datastream metadata announcement
KIND_DATASTREAM_DATA = 30101        # Observation data (encrypted DM)
KIND_SUBSCRIPTION_ANNOUNCE = 30102  # Subscription announcement
KIND_PAYMENT = 30103                # Payment notification (encrypted DM)


def compute_stream_topic_tag(stream_id: str) -> str:
    """Compute a standardized topic tag for a datastream.

    Used for filtering and discovery in Nostr relay queries.

    Args:
        stream_id: Stream identifier (e.g., "btc-price-usd")

    Returns:
        Topic tag string (e.g., "satori:stream:btc-price-usd")
    """
    return f"satori:stream:{stream_id}"
