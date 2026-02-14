# Stream Health Implementation Fix

## Problem with Original Design

The initial implementation had a **wasteful design flaw**:
- Stored `updated_at` in metadata (kind 30100 event)
- Auto-republished metadata every 10 observations just to update timestamp
- Unnecessary relay traffic and event storage

## The Better Solution

**Key insight:** Nostr event timestamps are already public!

Every Nostr event has public metadata separate from encrypted content:
```json
{
  "id": "event-id",
  "created_at": 1234567890,  // ← PUBLIC timestamp (always visible)
  "kind": 30101,
  "tags": [
    ["stream", "btc-price"],  // ← PUBLIC (visible to everyone)
    ["seq", "42"]             // ← PUBLIC
  ],
  "content": "encrypted..."    // ← ENCRYPTED (only recipient can read)
}
```

**So non-subscribers can:**
- Query relay for latest kind 30101 event for a stream
- See the event's `created_at` timestamp
- Know when last observation was published
- **Cannot** decrypt the actual observation data

## What Changed

### Removed
1. ❌ `updated_at` field from `DatastreamMetadata`
2. ❌ `_last_announce_times` tracking state
3. ❌ `_obs_counts` tracking state
4. ❌ `_should_update_announcement()` method
5. ❌ Auto-republishing logic in `publish_observation()`

### Added
1. ✅ `get_last_observation_time(stream_name)` - Queries relay for latest observation timestamp
2. ✅ Updated `is_likely_active(last_observation_time)` - Takes timestamp as parameter
3. ✅ Updated `discover_active_datastreams()` - Queries observation times from relay

### Kept
- ✅ `cadence_seconds` in metadata (static info, doesn't change often)
- ✅ Cadence constants (`CADENCE_HOURLY`, etc.)

## New API

### Check Stream Health

```python
# Get timestamp of last observation from relay
last_obs_time = await client.get_last_observation_time("btc-price")

if last_obs_time:
    # Check if stream is active
    is_active = metadata.is_likely_active(last_obs_time)

    # Or calculate age manually
    age_seconds = time.time() - last_obs_time
    print(f"Last observation was {age_seconds} seconds ago")
```

### Discover Active Streams

```python
# Automatically queries relay for observation times
active_streams = await client.discover_active_datastreams(
    tags=["bitcoin", "price"],
    max_staleness_multiplier=2.0
)

for stream in active_streams:
    last_time = await client.get_last_observation_time(stream.stream_name)
    print(f"{stream.name}: last observation at {last_time}")
```

## Example Usage

### Provider (Simple!)

```python
from satori_nostr import DatastreamMetadata, CADENCE_HOURLY

# Announce once with cadence
metadata = DatastreamMetadata(
    stream_name="btc-price",
    neuron_pubkey=client.pubkey(),
    name="Bitcoin Price Feed",
    description="Hourly BTC/USD",
    encrypted=True,
    price_per_obs=10,
    created_at=int(time.time()),
    cadence_seconds=CADENCE_HOURLY,  # Static info
    tags=["bitcoin", "price"]
)

await client.announce_datastream(metadata)

# Just publish observations - no auto-republishing!
for i in range(100):
    obs = DatastreamObservation(...)
    await client.publish_observation(obs, metadata)
```

### Subscriber

```python
# Find active streams (queries observation timestamps automatically)
active = await client.discover_active_datastreams(tags=["bitcoin"])

# Or check manually
metadata = await client.get_datastream("btc-price")
last_obs = await client.get_last_observation_time("btc-price")

if last_obs and metadata.is_likely_active(last_obs):
    print("Stream is active!")
```

## Why This is Better

1. **No Wasteful Re-publishing** - Metadata only published when it actually changes
2. **Uses Existing Infrastructure** - Event timestamps are already on relay
3. **More Accurate** - Shows actual last observation time, not last metadata update
4. **Simpler Provider Logic** - Providers just publish observations
5. **Public Visibility** - Non-subscribers can check stream health before subscribing

## How It Works

When subscriber checks if stream is active:

1. **Query relay:**
   ```python
   filter = Filter().kind(30101).custom_tag("stream", "btc-price").limit(1)
   events = await client.get_events_of([filter])
   ```

2. **Get timestamp from event metadata (public):**
   ```python
   last_obs_time = events[0].created_at().as_secs()
   ```

3. **Check against cadence:**
   ```python
   is_active = metadata.is_likely_active(last_obs_time)
   ```

The observation **content** is encrypted, but the event **timestamp** is always public!

## Migration

### Before (Wasteful)
```python
metadata = DatastreamMetadata(
    stream_name="btc-price",
    created_at=now,
    updated_at=now,  # ← Had to keep updating this
    cadence_seconds=3600,
    # ...
)

# Library auto-republished metadata every 10 observations
await client.publish_observation(obs, metadata)  # Triggers re-announce
```

### After (Efficient)
```python
metadata = DatastreamMetadata(
    stream_name="btc-price",
    created_at=now,
    cadence_seconds=3600,  # ← Static, doesn't change
    # ...
)

# No auto-republishing - just publish observation!
await client.publish_observation(obs, metadata)  # Clean and simple

# Subscribers query relay for observation timestamp
last_time = await client.get_last_observation_time("btc-price")
```

## Testing

All 21 model tests passing ✅
- Updated `is_likely_active()` tests to use timestamp parameter
- All examples updated
- All integration tests updated

## Files Modified

1. **models.py** - Removed `updated_at`, updated `is_likely_active()` signature
2. **client.py** - Removed tracking state, added `get_last_observation_time()`, updated `discover_active_datastreams()`
3. **tests/** - Updated all tests to remove `updated_at`
4. **examples/** - Updated all examples to remove `updated_at`

## Key Takeaway

**The timestamp is already in the Nostr event - we just query for it!**

No need to maintain separate tracking or republish metadata. The relay already stores the observation events with public timestamps.
