# Stream Health Tracking Feature

## Summary

Added stream health tracking to the Satori Nostr library to help subscribers identify which datastreams are actively publishing.

## New Fields in DatastreamMetadata

### `updated_at: int`
- Unix timestamp of the last published observation
- Updated automatically when provider publishes observations
- Used to determine if stream is still active

### `cadence_seconds: int | None`
- Expected seconds between observations
- `None` = irregular/event-driven streams
- Used to calculate if stream is stale

## Standard Cadence Constants

```python
CADENCE_REALTIME = 1       # Every second
CADENCE_MINUTE = 60        # Every minute
CADENCE_5MIN = 300         # Every 5 minutes
CADENCE_HOURLY = 3600      # Every hour (recommended)
CADENCE_DAILY = 86400      # Every day
CADENCE_WEEKLY = 604800    # Every week
CADENCE_IRREGULAR = None   # No fixed schedule
```

## Provider Behavior

### Automatic Announcement Updates

Providers automatically re-publish their announcement (kind 30100) periodically to update `updated_at`:

- **Every 10 observations**, OR
- **Every hour** (whichever is less frequent)

This uses Nostr's replaceable event feature (kind 30100) - relays automatically keep only the latest version.

### Implementation

```python
# Provider publishes observation
observation = DatastreamObservation(
    stream_name="btc-price",
    timestamp=int(time.time()),
    value={"price": 45000},
    seq_num=1
)

# Library automatically:
# 1. Sends observation to paid subscribers
# 2. Increments observation counter
# 3. Checks if announcement should be updated
# 4. Re-publishes announcement with new updated_at if needed

await client.publish_observation(observation, metadata)
```

## Subscriber APIs

### Check Individual Stream Health

```python
metadata = await client.get_datastream("btc-price")

# Check if stream is likely active
is_active = metadata.is_likely_active(max_staleness_multiplier=2.0)

# For regular cadence streams:
# - Active if: (now - updated_at) < (cadence_seconds * multiplier)
# - Example: Hourly stream (3600s), updated 1.5hrs ago, multiplier 2.0
#   → 5400 < 7200 → Active ✓

# For irregular cadence streams (None):
# - Active if updated in last 24 hours
```

### Discover Only Active Streams

```python
# Get only actively publishing streams
active_streams = await client.discover_active_datastreams(
    tags=["bitcoin", "price"],
    max_staleness_multiplier=2.0  # Allow up to 2x expected delay
)

for stream in active_streams:
    print(f"{stream.name}")
    print(f"  Last update: {stream.updated_at}")
    print(f"  Cadence: {stream.cadence_seconds}s")
```

## Example Usage

### Provider Setup

```python
from satori_nostr import (
    DatastreamMetadata,
    CADENCE_HOURLY
)

# Announce stream with cadence
metadata = DatastreamMetadata(
    stream_name="btc-price",
    neuron_pubkey=client.pubkey(),
    name="Bitcoin Price Feed",
    description="Hourly BTC/USD price",
    encrypted=True,
    price_per_obs=10,
    created_at=now,
    updated_at=now,
    cadence_seconds=CADENCE_HOURLY,  # Expect hourly updates
    tags=["bitcoin", "price"]
)

await client.announce_datastream(metadata)

# Publish observations - announcement auto-updates every 10 obs or hourly
for i in range(100):
    await client.publish_observation(observation, metadata)
```

### Subscriber Discovery

```python
# Find all Bitcoin streams
all_streams = await client.discover_datastreams(tags=["bitcoin"])

# Filter to only active ones
active_streams = [s for s in all_streams if s.is_likely_active()]

# Or use helper method
active_streams = await client.discover_active_datastreams(tags=["bitcoin"])

print(f"Found {len(active_streams)} active Bitcoin streams")
```

## Implementation Details

### Tracking State (Provider)

```python
# In SatoriNostr.__init__
self._last_announce_times: dict[str, int] = {}  # stream_name -> timestamp
self._obs_counts: dict[str, int] = {}  # stream_name -> count
```

### Update Logic

```python
def _should_update_announcement(self, metadata: DatastreamMetadata) -> bool:
    """Decide when to re-announce to update stream health."""
    obs_count = self._obs_counts.get(metadata.stream_name, 0)
    last_announce = self._last_announce_times.get(metadata.stream_name, 0)
    now = int(time.time())

    # Update every 10 observations
    if obs_count > 0 and obs_count % 10 == 0:
        return True

    # Or update hourly
    if (now - last_announce) > 3600:
        return True

    return False
```

### Staleness Check

```python
def is_likely_active(self, max_staleness_multiplier: float = 2.0) -> bool:
    """Check if stream appears to be actively publishing."""
    now = int(time.time())
    time_since_update = now - self.updated_at

    if self.cadence_seconds is None:
        # Irregular: active if updated in last 24 hours
        return time_since_update < 86400
    else:
        # Regular: check against expected cadence
        max_delay = self.cadence_seconds * max_staleness_multiplier
        return time_since_update < max_delay
```

## Benefits

1. **No Heartbeat Events** - Uses existing announcement mechanism (kind 30100)
2. **Relay Efficiency** - Replaceable events mean only 1 stored per stream
3. **Standard Protocol** - Works with any Nostr relay
4. **Flexible Discovery** - Subscribers can adjust staleness tolerance
5. **Automatic Updates** - Providers don't need to manually trigger health updates

## Testing

Added 2 new tests to `test_satori_nostr_models.py`:

- `test_is_likely_active_regular_cadence` - Tests staleness detection for regular cadence streams
- `test_is_likely_active_irregular_cadence` - Tests staleness detection for irregular streams

All 21 model tests passing ✅

## Migration

### Existing Code

Old code without new fields will need to add them:

```python
# Before
metadata = DatastreamMetadata(
    stream_name="btc-price",
    neuron_pubkey=pubkey,
    name="Bitcoin Price",
    description="BTC/USD",
    encrypted=True,
    price_per_obs=10,
    created_at=int(time.time()),
    tags=["bitcoin"]
)

# After
now = int(time.time())
metadata = DatastreamMetadata(
    stream_name="btc-price",
    neuron_pubkey=pubkey,
    name="Bitcoin Price",
    description="BTC/USD",
    encrypted=True,
    price_per_obs=10,
    created_at=now,
    updated_at=now,                    # NEW
    cadence_seconds=CADENCE_HOURLY,    # NEW
    tags=["bitcoin"]
)
```

## Files Modified

1. **models.py** - Added fields and `is_likely_active()` method
2. **client.py** - Added tracking and auto-update logic
3. **__init__.py** - Exported new constants
4. **test_satori_nostr_models.py** - Added tests, updated fixtures
5. **test_satori_nostr_integration.py** - Updated fixtures
6. **examples/simple_provider.py** - Added cadence
7. **examples/datastream_marketplace.py** - Added cadence

## Next Steps

Consider adding:
- UI to visualize stream health in a marketplace
- Metrics tracking (uptime %, missed observations)
- Alerts for stale streams subscribers depend on
- Historical health data storage
