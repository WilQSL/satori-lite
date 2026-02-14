# Satori Nostr Library

Datastream pub/sub with micropayments over Nostr relays.

## Overview

Satori Nostr enables neurons to publish and subscribe to datastreams with built-in micropayment support. Providers publish observations to paying subscribers via encrypted Nostr DMs, while maintaining public visibility of subscriptions for accountability.

## Features

- **Datastream Publishing** - Announce and publish data observations
- **Discovery** - Find datastreams by tags and topics
- **Encrypted Delivery** - NIP-04 encrypted unicast to each subscriber
- **Micropayments** - Pay-per-observation model
- **Public Accountability** - Subscription activity is publicly visible
- **Sparse Data Support** - Handle infrequent observations naturally

## Installation

```bash
cd /code/Satori/neuron
pip install nostr-sdk
```

## Quick Start

### Provider Example

```python
import asyncio
import time
from satori_nostr import (
    SatoriNostr,
    SatoriNostrConfig,
    DatastreamMetadata,
    DatastreamObservation,
)

async def provider_example():
    # Configure client
    config = SatoriNostrConfig(
        keys="nsec1...",  # Your Nostr private key
        relay_urls=["wss://relay.damus.io", "wss://nostr.wine"]
    )

    client = SatoriNostr(config)
    await client.start()

    # Announce a datastream
    metadata = DatastreamMetadata(
        stream_id="btc-price-usd",
        neuron_pubkey=client.pubkey(),
        name="Bitcoin Price (USD)",
        description="Real-time BTC/USD from Coinbase",
        encrypted=True,
        price_per_obs=10,  # 10 sats per observation
        created_at=int(time.time()),
        tags=["bitcoin", "price", "usd"]
    )

    await client.announce_datastream(metadata)
    print("Datastream announced!")

    # Publish observations to paid subscribers
    for seq in range(1, 100):
        observation = DatastreamObservation(
            stream_id="btc-price-usd",
            timestamp=int(time.time()),
            value={"price": 45000 + seq * 100, "volume": 123.45},
            seq_num=seq
        )

        event_ids = await client.publish_observation(observation, metadata)
        print(f"Observation {seq} sent to {len(event_ids)} subscribers")

        await asyncio.sleep(60)  # Once per minute

    await client.stop()

asyncio.run(provider_example())
```

### Subscriber Example

```python
import asyncio
from satori_nostr import SatoriNostr, SatoriNostrConfig

async def subscriber_example():
    # Configure client
    config = SatoriNostrConfig(
        keys="nsec1...",  # Your Nostr private key
        relay_urls=["wss://relay.damus.io"]
    )

    client = SatoriNostr(config)
    await client.start()

    # Discover datastreams
    streams = await client.discover_datastreams(tags=["bitcoin"])
    print(f"Found {len(streams)} Bitcoin datastreams")

    # Subscribe to a stream
    if streams:
        stream = streams[0]
        await client.subscribe_datastream(
            stream.stream_id,
            stream.neuron_pubkey,
            payment_channel="lightning:channel123"
        )
        print(f"Subscribed to {stream.name}")

        # Receive observations
        obs_count = 0
        async for inbound in client.observations():
            obs = inbound.observation
            print(f"Observation {obs.seq_num}: {obs.value}")

            # Send payment for this observation
            await client.send_payment(
                provider_pubkey=stream.neuron_pubkey,
                stream_id=stream.stream_id,
                seq_num=obs.seq_num + 1,  # Pay for NEXT observation
                amount_sats=stream.price_per_obs
            )

            obs_count += 1
            if obs_count >= 10:
                break

    await client.stop()

asyncio.run(subscriber_example())
```

## Architecture

### Event Kinds

| Kind | Name | Encryption | Purpose |
|------|------|-----------|----------|
| 30100 | Datastream Announce | Public | Stream metadata discovery |
| 30101 | Datastream Data | Encrypted DM | Observation delivery |
| 30102 | Subscription Announce | Public | Subscription visibility |
| 30103 | Payment | Encrypted DM | Micropayment notifications |

### Payment Flow

```
1. Subscriber discovers stream (query kind 30100)
2. Subscriber announces subscription (publish kind 30102, public)
3. Provider sees subscription, adds to subscriber list
4. Provider sends first observation FREE (signals acceptance)
5. For each subsequent observation:
   a. Subscriber sends payment (kind 30103, encrypted DM)
   b. Provider receives payment, updates subscriber state
   c. Provider publishes observation (kind 30101, encrypted DM to each paid subscriber)
   d. Subscriber receives observation
```

### Data Models

**DatastreamMetadata** (kind 30100 content)
```python
{
    "stream_id": "btc-price-usd",
    "neuron_pubkey": "abc123...",
    "name": "Bitcoin Price (USD)",
    "description": "Real-time BTC/USD",
    "encrypted": true,
    "price_per_obs": 10,
    "created_at": 1234567890,
    "tags": ["bitcoin", "price"]
}
```

**DatastreamObservation** (kind 30101 encrypted content)
```python
{
    "stream_id": "btc-price-usd",
    "timestamp": 1234567890,
    "value": {"price": 45000, "volume": 123.45},
    "seq_num": 42
}
```

**PaymentNotification** (kind 30103 encrypted content)
```python
{
    "from_pubkey": "subscriber...",
    "to_pubkey": "provider...",
    "stream_id": "btc-price-usd",
    "seq_num": 43,
    "amount_sats": 10,
    "timestamp": 1234567890,
    "tx_id": "lightning:tx789"
}
```

## API Reference

### SatoriNostr

Main client class for datastream pub/sub.

#### Configuration

```python
config = SatoriNostrConfig(
    keys="nsec1..." or "hex_private_key",
    relay_urls=["wss://relay.damus.io"],
    active_relay_timeout_ms=8000,  # Optional
    dedupe_db_path=None  # Optional: SQLite path
)
```

#### Provider Methods

- `announce_datastream(metadata) -> str` - Publish stream metadata
- `publish_observation(obs, metadata) -> list[str]` - Send obs to paid subscribers
- `get_subscribers(stream_id) -> list[str]` - List subscriber pubkeys
- `record_subscription(stream_id, sub_pubkey, channel)` - Record new subscriber
- `record_payment(stream_id, sub_pubkey, seq_num)` - Record payment
- `payments() -> AsyncIterator[InboundPayment]` - Receive payment notifications

#### Subscriber Methods

- `discover_datastreams(tags, limit) -> list[DatastreamMetadata]` - Find streams
- `subscribe_datastream(stream_id, provider_pk, channel) -> str` - Subscribe
- `send_payment(provider_pk, stream_id, seq, amount, tx_id) -> str` - Send payment
- `observations() -> AsyncIterator[InboundObservation]` - Receive observations

#### Lifecycle

- `start()` - Connect to relays and start processing
- `stop()` - Disconnect and cleanup
- `pubkey() -> str` - Get client's public key
- `is_running() -> bool` - Check if active

## Design Principles

1. **Encrypted Unicast** - Each observation sent as encrypted DM to each subscriber
2. **Public Accountability** - Subscriptions and metadata are publicly visible
3. **Micropayments** - Pay-per-observation model with Lightning support
4. **Sparse Data** - Designed for infrequent observations (hours/days between data)
5. **Simple P2P** - Direct provider → subscriber relationship over Nostr relays

## Cost Model

For sparse streams:
- 100 subscribers × 24 observations/day = 2,400 events/day (~2.4 MB)
- Very affordable for hourly/daily datastreams

For high-frequency streams, consider batching payments (e.g., pay for 100 observations at once).

## Security

- **NIP-04 Encryption** - Observations and payments use standard Nostr encryption
- **No Relay Trust** - Relays are dumb pipes; they cannot read encrypted content
- **Access Control** - Providers control who receives observations based on payment state

## Future Enhancements

- [ ] NIP-17/59 (Gift-wrapped events) for better privacy
- [ ] Shared key optimization for high-frequency streams
- [ ] Historical data requests with batch payment
- [ ] Reputation scoring for providers
- [ ] Multi-currency payment support

## License

MIT License - See LICENSE file for details

## Version

Current version: **0.1.0** (MVP)

Nostr Implementation:
- NIP-04: Encrypted Direct Messages ✅
- Custom kinds: 30100-30103 ✅
