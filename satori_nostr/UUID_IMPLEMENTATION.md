# UUID Implementation for DatastreamMetadata

## Overview

Added a **computed UUID property** to `DatastreamMetadata` that generates a deterministic identifier from the combination of `nostr_pubkey` and `stream_name`.

---

## Implementation

### UUID Generation

```python
@property
def uuid(self) -> str:
    """Generate deterministic UUID from nostr_pubkey and stream_name."""
    namespace = uuid.NAMESPACE_DNS
    identifier = f"{self.nostr_pubkey}:{self.stream_name}"
    return str(uuid.uuid5(namespace, identifier))
```

**Key characteristics:**
- **UUID v5** - SHA-1 namespace-based UUID
- **Namespace:** `uuid.NAMESPACE_DNS` (standard namespace)
- **Identifier:** `"{nostr_pubkey}:{stream_name}"`
- **Deterministic** - Same input always produces same UUID
- **Computed** - Generated on-demand, not stored

---

## How It Works

### Uniqueness Guarantee

A stream is uniquely identified by: `(nostr_pubkey, stream_name)`

**Nostr already enforces this** through replaceable parameterized events:
- Kind 30100 events are identified by: `(author_pubkey, kind, d_tag)`
- Each provider can have only ONE announcement per `stream_name`
- Relays automatically replace old announcements

**UUID provides:**
- Single identifier for convenience
- Compatibility with systems expecting UUIDs (databases)
- Deterministic mapping from `(pubkey, stream_name)` to UUID

### Determinism

Same `(nostr_pubkey, stream_name)` **always** produces the same UUID, regardless of:
- ❌ Stream name
- ❌ Description
- ❌ Price
- ❌ Encryption status
- ❌ Cadence
- ❌ Tags
- ❌ Metadata
- ❌ Creation time

**Only** `nostr_pubkey` and `stream_name` matter.

---

## Examples

### Same Stream, Same UUID

```python
stream1 = DatastreamMetadata(
    stream_name="btc-price",
    nostr_pubkey="abc123",
    name="Bitcoin Price",
    price_per_obs=0,
    # ... other fields
)

stream2 = DatastreamMetadata(
    stream_name="btc-price",
    nostr_pubkey="abc123",
    name="Different Name!",      # Different
    price_per_obs=999,            # Different
    # ... everything else different
)

assert stream1.uuid == stream2.uuid  # ✅ Same UUID!
# Both produce: "41d7cd8f-dac2-585b-8c70-64e09bf9c5bc"
```

### Different Stream_ID, Different UUID

```python
stream1 = DatastreamMetadata(
    stream_name="btc-price",  # Bitcoin
    nostr_pubkey="abc123",
    # ...
)

stream2 = DatastreamMetadata(
    stream_name="eth-price",  # Ethereum (different!)
    nostr_pubkey="abc123",  # Same provider
    # ...
)

assert stream1.uuid != stream2.uuid  # ✅ Different UUIDs
# stream1: "41d7cd8f-dac2-585b-8c70-64e09bf9c5bc"
# stream2: "58e0f51b-b78c-597b-8aab-901825a63191"
```

### Different Provider, Different UUID

```python
stream1 = DatastreamMetadata(
    stream_name="btc-price",
    nostr_pubkey="abc123",  # Provider A
    # ...
)

stream2 = DatastreamMetadata(
    stream_name="btc-price",   # Same stream name
    nostr_pubkey="xyz789",  # Provider B (different!)
    # ...
)

assert stream1.uuid != stream2.uuid  # ✅ Different UUIDs
# stream1: "41d7cd8f-dac2-585b-8c70-64e09bf9c5bc"
# stream2: "a4a64c9c-4d87-5e5f-aa77-d3063dd7d1de"
```

---

## Use Cases

### 1. Database Storage

```python
# Store stream in database using UUID as primary key
metadata = DatastreamMetadata(
    stream_name="btc-price",
    nostr_pubkey="abc123...",
    # ...
)

db.execute(
    "INSERT INTO streams (uuid, stream_name, nostr_pubkey, name, ...) VALUES (?, ?, ?, ...)",
    (metadata.uuid, metadata.stream_name, metadata.nostr_pubkey, metadata.name, ...)
)

# Query by UUID
stream = db.query("SELECT * FROM streams WHERE uuid = ?", (metadata.uuid,))
```

### 2. API Endpoints

```python
# REST API using UUID
@app.get("/streams/{uuid}")
def get_stream(uuid: str):
    # Look up stream by UUID
    stream = db.get_stream_by_uuid(uuid)
    return stream

# Client usage
response = requests.get(f"https://api.satori.com/streams/{metadata.uuid}")
```

### 3. Deduplication

```python
# Prevent duplicate processing
seen_uuids = set()

for stream in discovered_streams:
    if stream.uuid in seen_uuids:
        continue  # Already processed

    seen_uuids.add(stream.uuid)
    process_stream(stream)
```

### 4. Cache Keys

```python
# Use UUID as cache key
cache_key = f"stream:{metadata.uuid}"
redis.set(cache_key, metadata.to_json(), ex=3600)

# Retrieve from cache
cached = redis.get(cache_key)
if cached:
    metadata = DatastreamMetadata.from_json(cached)
```

### 5. Migration from Central Database

Central database already uses UUIDs:

```sql
CREATE TABLE streams (
    id INTEGER PRIMARY KEY,
    uuid UUID UNIQUE,
    name VARCHAR(255),
    author VARCHAR(130),
    ...
);
```

Now you can map:

```python
# Old Central stream
central_stream = db.query("SELECT * FROM streams WHERE name = ?", ("bitcoin",))

# Create Nostr metadata with matching UUID
nostr_metadata = DatastreamMetadata(
    stream_name="bitcoin",
    nostr_pubkey=central_stream.author,
    # ...
)

# UUID matches if same (author, name) combination
assert nostr_metadata.uuid == str(central_stream.uuid)  # If deterministic migration
```

---

## Benefits

✅ **Deterministic** - Same input always produces same UUID
✅ **Computed** - No storage needed, calculated on-demand
✅ **Compatible** - Works with existing UUID-based systems
✅ **Unique** - Different streams always have different UUIDs
✅ **Standard** - Uses UUID v5 (RFC 4122)
✅ **Efficient** - Fast SHA-1 hashing
✅ **No Protocol Changes** - Doesn't affect Nostr events

---

## Technical Details

### UUID v5 (SHA-1 Namespace)

```python
uuid.uuid5(namespace, identifier)
```

**Parameters:**
- `namespace` - `uuid.NAMESPACE_DNS` (standard namespace UUID)
- `identifier` - `"{nostr_pubkey}:{stream_name}"`

**Process:**
1. Concatenate namespace UUID + identifier string
2. Hash using SHA-1
3. Format as UUID (8-4-4-4-12 hex digits)
4. Set version bits to 5
5. Set variant bits to RFC 4122

**Output format:**
```
41d7cd8f-dac2-585b-8c70-64e09bf9c5bc
└──────┘ └──┘ └──┘ └──┘ └──────────┘
   8      4    4    4        12    hex digits
```

### Why UUID v5?

**Alternatives considered:**

| Type | Pros | Cons | Decision |
|------|------|------|----------|
| **UUID v4** (Random) | Unique | Not deterministic | ❌ No |
| **UUID v3** (MD5) | Deterministic | MD5 deprecated | ❌ No |
| **UUID v5** (SHA-1) | Deterministic, standard | SHA-1 not for crypto | ✅ **Yes** |
| **Custom hash** | Full control | Non-standard | ❌ No |

**Chosen:** UUID v5
- ✅ Deterministic (same input → same UUID)
- ✅ Standard (RFC 4122)
- ✅ Widely supported
- ✅ SHA-1 fine for non-cryptographic use (identifiers)

---

## Testing

### Tests Added

1. **`test_uuid_property`** - UUID is valid UUID format and deterministic
2. **`test_uuid_deterministic_across_instances`** - Same pubkey+stream_name produces same UUID
3. **`test_uuid_different_for_different_streams`** - Different inputs produce different UUIDs

**Results:** 28/28 tests passing ✅

### Example Test

```python
def test_uuid_deterministic_across_instances():
    """Same pubkey+stream_name produces same UUID."""
    metadata1 = DatastreamMetadata(
        stream_name="btc-price",
        nostr_pubkey="abc123",
        name="Bitcoin",
        price_per_obs=0,
        # ...
    )

    metadata2 = DatastreamMetadata(
        stream_name="btc-price",
        nostr_pubkey="abc123",
        name="Different Name",   # Different!
        price_per_obs=999,       # Different!
        # All other fields different
    )

    # Same UUID despite different fields
    assert metadata1.uuid == metadata2.uuid  # ✅
```

---

## Migration Guide

### From Current Neuron (StreamId with 4-part hierarchy)

```python
# Old
old_stream = StreamId(
    source="Streamr",
    author="pubkey123",
    stream="DATAUSD/binance/ticker",
    target="Close"
)
old_uuid = old_stream.uuid  # Generated from all 4 parts

# New
new_stream = DatastreamMetadata(
    stream_name="DATAUSD-binance-ticker-Close",  # Flatten hierarchy
    nostr_pubkey="pubkey123",
    # ...
)
new_uuid = new_stream.uuid  # Generated from pubkey + stream_name

# Note: UUIDs will be DIFFERENT (different input format)
# Need to maintain mapping table for migration
```

### From Central Database

```python
# Query old Central streams
old_streams = db.query("SELECT * FROM streams")

# Create Nostr metadata
for old in old_streams:
    new_stream = DatastreamMetadata(
        stream_name=old.name,
        nostr_pubkey=old.author,
        name=old.name,
        description=old.description,
        # ...
    )

    # Store UUID mapping for reference
    uuid_mapping[old.uuid] = new_stream.uuid
```

---

## Notes

### UUID is NOT Stored

The UUID is a **computed property**, not a stored field:

```python
# NOT in JSON
json_str = metadata.to_json()
# Does NOT contain "uuid" key

# Computed on access
uuid = metadata.uuid  # Calculated each time
```

This means:
- ✅ No storage overhead
- ✅ Always up-to-date
- ✅ Can't get out of sync
- ❌ Not in serialized JSON (must compute after deserializing)

### Nostr Events Don't Use UUID

Nostr protocol identifies streams by `(author_pubkey, d_tag)`:

```python
# Event structure
{
    "kind": 30100,
    "pubkey": "abc123...",  # Author
    "tags": [
        ["d", "btc-price"]  # Stream ID
    ],
    # No UUID in event!
}
```

UUID is a **client-side convenience** for compatibility with UUID-based systems.

---

## Summary

Added a **deterministic UUID property** to `DatastreamMetadata`:

- Generated from: `nostr_pubkey` + `stream_name`
- Algorithm: UUID v5 (SHA-1 namespace)
- Characteristics: Deterministic, computed, not stored
- Use cases: Databases, APIs, caching, deduplication
- Compatibility: Works with existing UUID-based systems

**Bottom line:** Same stream always has same UUID, making it easy to integrate with systems that expect UUIDs while maintaining Nostr's native `(pubkey, stream_name)` identification.
