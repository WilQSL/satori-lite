"""Satori Nostr - Datastream pub/sub with micropayments over Nostr."""

__version__ = "0.1.0"

from .client import SatoriNostr, SubscriberState
from .models import (
    DatastreamMetadata,
    DatastreamObservation,
    SubscriptionAnnouncement,
    PaymentNotification,
    InboundObservation,
    InboundPayment,
    SatoriNostrConfig,
    KIND_DATASTREAM_ANNOUNCE,
    KIND_DATASTREAM_DATA,
    KIND_SUBSCRIPTION_ANNOUNCE,
    KIND_PAYMENT,
    compute_stream_topic_tag,
)
from .dedupe import DedupeCache
from .encryption import (
    encrypt_json,
    decrypt_json,
    encrypt_observation,
    decrypt_observation,
    encrypt_payment,
    decrypt_payment,
    EncryptionError,
)

__all__ = [
    # Main Client
    "SatoriNostr",
    "SubscriberState",

    # Models
    "DatastreamMetadata",
    "DatastreamObservation",
    "SubscriptionAnnouncement",
    "PaymentNotification",
    "InboundObservation",
    "InboundPayment",
    "SatoriNostrConfig",

    # Constants
    "KIND_DATASTREAM_ANNOUNCE",
    "KIND_DATASTREAM_DATA",
    "KIND_SUBSCRIPTION_ANNOUNCE",
    "KIND_PAYMENT",

    # Functions
    "compute_stream_topic_tag",

    # Utilities
    "DedupeCache",
    "encrypt_json",
    "decrypt_json",
    "encrypt_observation",
    "decrypt_observation",
    "encrypt_payment",
    "decrypt_payment",
    "EncryptionError",
]
