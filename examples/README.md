# Satori Nostr Examples

Example applications demonstrating Satori Nostr library usage.

## Examples

### 1. Simple Provider (`simple_provider.py`)

Demonstrates how to run a basic datastream provider:
- Announces a Bitcoin price datastream
- Accepts subscriptions
- Monitors payment notifications
- Publishes observations every 30 seconds

**Run:**
```bash
cd /code/Satori/neuron
python examples/simple_provider.py
```

### 2. Simple Subscriber (`simple_subscriber.py`)

Demonstrates how to consume datastreams:
- Discovers available datastreams
- Subscribes to a stream
- Receives observations
- Sends micropayments

**Run:**
```bash
cd /code/Satori/neuron
python examples/simple_subscriber.py
```

### 3. Datastream Marketplace (`datastream_marketplace.py`)

Complete ecosystem simulation with:
- 3 providers offering different datastreams:
  - Bitcoin Price Feed (10 sats/obs)
  - Weather Station NYC (free)
  - Tech News Feed (5 sats/obs)
- 3 subscribers consuming different data types
- Payment flows and access control
- Both free and paid streams

**Run:**
```bash
cd /code/Satori/neuron
python examples/datastream_marketplace.py
```

**Output example:**
```
DATASTREAM MARKETPLACE SIMULATION
============================================================

PROVIDERS STARTING...
------------------------------------------------------------
✓ Bitcoin Price Feed online (Provider: abc123...)
✓ Weather Station NYC online (Provider: def456...)
✓ Tech News Feed online (Provider: ghi789...)

SUBSCRIBERS STARTING...
------------------------------------------------------------
✓ Crypto Trader Bot online (Subscriber: jkl012...)
✓ Weather Analytics online (Subscriber: mno345...)
✓ News Aggregator online (Subscriber: pqr678...)

DISCOVERY & SUBSCRIPTION...
------------------------------------------------------------
  Crypto Trader Bot found 1 stream(s) with tags ['bitcoin', 'price']
  Crypto Trader Bot subscribed to Bitcoin Price Feed
  Crypto Trader Bot sent payment for Bitcoin Price Feed
  ...

DATASTREAM ACTIVITY...
------------------------------------------------------------
[Round 1]
  Bitcoin Price Feed published → 1 subscriber(s)
  💰 Bitcoin Price Feed received 10 sats for seq #1
  📊 Crypto Trader Bot received: Bitcoin Price Feed seq #1
  ...
```

## Testing with Real Relays

All examples connect to public Nostr relays:
- `wss://relay.damus.io`
- `wss://nostr.wine`
- `wss://relay.primal.net`

Your data will be visible on the public Nostr network!

## Using Your Own Keys

For production use, provide your own Nostr keys via environment variables:

```bash
export NOSTR_PROVIDER_KEY="nsec1..."
export NOSTR_SUBSCRIBER_KEY="nsec1..."
python examples/simple_provider.py
```

Or modify the examples to load keys from a file:

```python
from nostr_sdk import Keys

# Load from file
with open("provider_key.txt") as f:
    keys = Keys.parse(f.read().strip())
```

## Payment Integration

The examples use placeholder payment transaction IDs (`demo:tx_123`). In production, integrate with:

- **Lightning Network**: Use LND, CLN, or LNBits
- **On-chain**: Use Bitcoin Core RPC or electrum
- **Payment channels**: Set up dedicated channels between providers/subscribers

Example with Lightning:

```python
# After receiving observation, pay via Lightning
payment_hash = await lightning_client.pay_invoice(provider_invoice)

# Send payment notification with proof
await client.send_payment(
    provider_pubkey=provider_pk,
    stream_id=stream_id,
    seq_num=next_seq,
    amount_sats=price,
    tx_id=f"lightning:{payment_hash}"
)
```

## Running Multiple Instances

You can run multiple providers and subscribers simultaneously:

**Terminal 1:**
```bash
python examples/simple_provider.py
```

**Terminal 2:**
```bash
python examples/simple_subscriber.py
```

**Terminal 3:**
```bash
python examples/datastream_marketplace.py
```

They'll all discover each other through the relay network!

## Customization

### Adding Custom Datastreams

Modify the provider examples to publish your own data:

```python
observation = DatastreamObservation(
    stream_id="my-custom-stream",
    timestamp=int(time.time()),
    value={
        "temperature": sensor.read_temperature(),
        "humidity": sensor.read_humidity(),
        "location": "rooftop_sensor_01"
    },
    seq_num=seq_num
)
```

### Custom Discovery

Filter datastreams by custom criteria:

```python
# Discover by tags
streams = await client.discover_datastreams(
    tags=["weather", "temperature"],
    limit=50
)

# Filter by price
free_streams = [s for s in streams if s.price_per_obs == 0]
affordable_streams = [s for s in streams if s.price_per_obs <= 100]

# Filter by provider reputation (you'd track this)
trusted_streams = [s for s in streams if s.neuron_pubkey in trusted_providers]
```

## Next Steps

After running the examples:

1. **Build your own datastream** - Publish real sensor data, API feeds, or ML predictions
2. **Create a marketplace UI** - Build a web interface for discovery and management
3. **Integrate payments** - Connect real Lightning Network payments
4. **Add monitoring** - Track stream health, payment success rates, subscriber counts
5. **Scale up** - Deploy multiple providers across different geographic regions

## Troubleshooting

**No streams discovered:**
- Make sure provider is running first
- Wait 10-30 seconds for relay propagation
- Check relay connectivity

**Observations not received:**
- Verify subscription was announced
- Check payment was sent (for paid streams)
- Ensure provider recorded your subscription
- Check Nostr client logs

**Payment issues:**
- Verify payment event was published
- Check provider is monitoring payment events
- Ensure correct stream_id and seq_num in payment

## Learn More

- See `/code/Satori/neuron/satori_nostr/README.md` for full API documentation
- Check `/code/docs/design/satori_nostr_adaptation.md` for design details
- Review tests in `/code/Satori/neuron/tests/test_satori_nostr_*.py`
