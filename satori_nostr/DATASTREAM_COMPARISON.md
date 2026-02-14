# Datastream Comparison: Current vs Nostr Implementation

Comprehensive comparison of the existing datastream implementations (Neuron and Central) with the new Nostr-based implementation.

---

## 1. IDENTIFICATION FIELDS

### Current Implementation (Neuron `Stream` class)

**StreamId** - Complex hierarchical identifier with 4 parts:
- `source: str` - Data source (e.g., "Streamr", "satori")
- `author: str` - Wallet pubkey of provider
- `stream: str` - Stream name (e.g., "bitcoin", "WeatherBerlin")
- `target: str` - Target field within data (e.g., "Close", "temperature")

**Derived identifiers:**
- `uuid: str` - Generated via `uuid.uuid5()` from the 4-part ID
- `id: tuple` - The 4-part tuple
- `topic: str` - JSON representation: `{"source": "...", "author": "...", "stream": "...", "target": "..."}`
- `mapId: dict` - Dictionary version
- `jsonId: str` - JSON string version

**Example:**
```python
StreamId(
    source="Streamr",
    author="02a85fb71485c6d7c62a3784c5549bd3849d0afa3ee44ce3f9ea5541e4c56402d8",
    stream="WeatherBerlin",
    target="temperature"
)
# UUID: generated from all 4 parts
# Topic: '{"source": "Streamr", "author": "02a85...", "stream": "WeatherBerlin", "target": "temperature"}'
```

### Current Implementation (Central Database)

**Stream table fields:**
- `id: int` - Auto-incrementing primary key
- `uuid: UUID` - Unique identifier (PostgreSQL UUID type)
- `name: str` - Stream name (required, max 255 chars)
- `author: str` - Wallet pubkey (optional, max 130 chars)
- `secondary: str` - Optional secondary identifier (max 255 chars)
- `target: str` - Optional target field (max 255 chars)
- `meta: str` - Optional metadata (max 255 chars)

**Notes:**
- Central uses simpler flat structure
- UUID is auto-generated, not derived from components
- `source` is not a separate field (may be in `secondary` or `meta`)

### Nostr Implementation

**Simplified identifier:**
- `stream_name: str` - Simple string identifier (e.g., "btc-price-usd", "weather-nyc")
- `nostr_pubkey: str` - Provider's Nostr public key (hex)

**How identification works:**
- Stream uniquely identified by: `(stream_name, nostr_pubkey)` pair
- No hierarchical structure (source/target)
- Nostr event kind differentiates stream types (via tags)
- Tags provide categorization instead of hierarchical ID

**Example:**
```python
DatastreamMetadata(
    stream_name="btc-price-usd",  # Simple string
    nostr_pubkey="abc123...",  # Provider pubkey
    tags=["bitcoin", "price", "usd"]  # Categorization via tags
)
```

---

## 2. METADATA FIELDS

### Current Implementation (Neuron `Stream` class)

**Core metadata:**
- `cadence: int` - Expected seconds between observations (optional)
- `offset: int` - UTC offset (optional, unused)
- `datatype: str` - Data type (optional, unused)
- `description: str` - Human-readable description (optional, unused)
- `tags: str` - Tags as string (optional, unused)
- `price_per_obs: float` - Price in satoshis (optional)

**Source/API fields:**
- `url: str` - Source URL (optional, unused)
- `uri: str` - URI identifier (optional)
- `headers: str` - HTTP headers for API calls (optional)
- `payload: str` - HTTP payload (optional)
- `hook: str` - Webhook endpoint (optional)

**Prediction fields:**
- `predicting: StreamId` - What this stream predicts (optional)
- `reason: StreamId` - Why subscribing (for prediction) (optional)
- `reason_is_primary: bool` - Is this the primary reason? (optional)

**Other:**
- `history: str` - Historical data info (optional)
- `ts: str` - Timestamp (optional)
- `pinned: int` - Is stream pinned? (optional)
- `kwargs: dict` - Additional keyword arguments

**Notes:**
- Many fields marked "unused" in code comments
- Designed for both raw data and prediction streams
- Mixed concerns (data source config + metadata + predictions)

### Current Implementation (Central Database)

**Fields:**
- `name: str` - Required stream name
- `author: str` - Wallet pubkey (nullable)
- `secondary: str` - Optional secondary identifier (nullable)
- `target: str` - Optional target field (nullable)
- `meta: str` - Optional metadata string (nullable)
- `description: Text` - Optional description (nullable, unlimited length)
- `ts: DateTime` - Auto-generated creation timestamp

**Notes:**
- Much simpler than Neuron
- No cadence, price, or prediction fields
- Database normalized (separate observations table)

### Nostr Implementation

**All fields:**
- `stream_name: str` - Identifier
- `nostr_pubkey: str` - Provider public key
- `name: str` - Human-readable name
- `description: str` - What this stream provides
- `encrypted: bool` - Is data encrypted? (True for paid streams)
- `price_per_obs: int` - Price in satoshis (0 = free)
- `created_at: int` - Unix timestamp of creation
- `cadence_seconds: int | None` - Expected seconds between observations (None = irregular)
- `tags: list[str]` - Searchable tags for discovery

**Notes:**
- Focused, minimal design
- All fields actively used
- Tags are proper list, not string
- No unused fields
- Stream health tracked via observation timestamps (not metadata)

---

## 3. CATEGORIZATION & DISCOVERY

### Current Implementation (Neuron)

**Hierarchical categorization:**
- Fixed 4-level hierarchy: `source → author → stream → target`
- No flexible tagging system
- Discovery requires knowing hierarchy structure

**Example:**
```python
StreamId(
    source="Streamr",           # Level 1
    author="pubkey...",          # Level 2
    stream="DATAUSD/binance/ticker",  # Level 3
    target="Close"               # Level 4
)
```

**Limitations:**
- Rigid structure
- Can't easily search by arbitrary categories
- Complex to navigate

### Current Implementation (Central)

**Simple structure:**
- Single `name` field
- No built-in categorization
- Would need to parse name or use `meta` field for categories

**Discovery:**
- Query by name (exact match)
- No tag-based discovery
- Would require full-text search or manual parsing

### Nostr Implementation

**Tag-based categorization:**
- Flexible list of tags: `["bitcoin", "price", "usd", "prediction"]`
- Can add any number of tags
- Easy to search/filter

**Discovery:**
```python
# Find all Bitcoin-related streams
streams = await client.discover_datastreams(tags=["bitcoin"])

# Find Bitcoin predictions
predictions = await client.discover_datastreams(tags=["bitcoin", "prediction"])

# Find free weather streams
free_weather = await client.discover_datastreams(tags=["weather", "free"])
```

**Advantages:**
- Flexible categorization
- Easy discovery
- Can filter by multiple criteria
- Standard Nostr relay queries

---

## 4. PRICING & PAYMENTS

### Current Implementation (Neuron)

**Field:**
- `price_per_obs: float` - Price in satoshis

**Implementation:**
- Field exists but no payment flow implemented
- Would need separate payment system
- No built-in micropayment protocol

### Current Implementation (Central)

**No pricing fields:**
- Database has no price information
- No payment tracking
- Would need to add separate tables for payments

### Nostr Implementation

**Built-in payment flow:**
- `price_per_obs: int` - Price in satoshis (0 = free)
- `encrypted: bool` - Ties to pricing (paid streams encrypted)
- Payment notifications via kind 30103 events
- Automatic access control based on payments
- Subscriber state tracking

**Payment flow:**
```python
# Announce paid stream
metadata = DatastreamMetadata(
    price_per_obs=10,  # 10 sats per observation
    encrypted=True
)

# Subscriber pays
await client.send_payment(
    provider_pubkey=pk,
    stream_name="btc-price",
    seq_num=5,
    amount_sats=10
)

# Provider tracks payments automatically
# Only sends to subscribers who paid
```

**Advantages:**
- Integrated payment protocol
- Automatic access control
- Public payment visibility (accountability)
- Micropayment-friendly

---

## 5. DATA DELIVERY

### Current Implementation (Neuron)

**Pubsub via Centrifugo:**
- Topic-based subscription
- Centralized message broker
- Observations delivered as JSON over WebSocket

**Format:**
```json
{
  "topic": "{\"source\": \"satori\", \"author\": \"pubkey\", \"stream\": \"WeatherBerlin\", \"target\": \"temperature\"}",
  "time": "2024-04-13 17:53:00.661619",
  "data": 4.2,
  "hash": "abc"
}
```

**Characteristics:**
- Real-time delivery
- Requires persistent connection
- Centralized infrastructure
- No built-in encryption

### Current Implementation (Central)

**HTTP REST API:**
- POST observations to Central server
- Stored in database
- Queried via GET endpoints
- Not real-time (polling needed)

### Nostr Implementation

**Decentralized relay network:**
- Encrypted DMs (kind 30101) per subscriber
- Public relay infrastructure
- Automatic deduplication
- Failover between relays

**Format:**
```json
{
  "kind": 30101,
  "content": "encrypted_observation_json",
  "tags": [
    ["p", "subscriber_pubkey"],
    ["stream", "btc-price"],
    ["seq", "42"]
  ],
  "created_at": 1234567890
}
```

**Characteristics:**
- Encrypted by default (for paid streams)
- Decentralized (no single point of failure)
- Public accountability (event timestamps visible)
- Works with any Nostr relay

---

## 6. STREAM HEALTH / ACTIVITY TRACKING

### Current Implementation (Neuron)

**Field:**
- `cadence: int` - Expected seconds between observations

**Usage:**
- Field exists but not actively used for health checks
- Would need to query observations to check activity
- No standard way to determine if stream is alive

### Current Implementation (Central)

**No health tracking:**
- `ts` field shows when stream was created
- No field for last observation time
- Would need to query observations table

### Nostr Implementation

**Cadence + observation timestamps:**
- `cadence_seconds: int | None` - Expected frequency
- Observations have public timestamps (Nostr event `created_at`)
- Active health checking via relay queries

**Health check:**
```python
# Get last observation timestamp from relay
last_obs_time = await client.get_last_observation_time("btc-price")

# Check if stream is active
is_active = metadata.is_likely_active(
    last_obs_time,
    max_staleness_multiplier=2.0
)

# Discover only active streams
active = await client.discover_active_datastreams(tags=["bitcoin"])
```

**Advantages:**
- No metadata republishing needed
- Observation timestamps already on relay
- Easy to check stream health before subscribing
- Standard across all streams

---

## 7. PREDICTIONS

### Current Implementation (Neuron)

**Built-in prediction support:**
- `predicting: StreamId` - Stream being predicted
- `reason: StreamId` - Stream subscribed to for prediction
- `reason_is_primary: bool` - Primary prediction reason?

**Design:**
- Predictions are streams themselves
- Complex relationships between data and prediction streams
- `StreamPair` class matches subscriptions to publications

**Example:**
```python
# Subscribe to Bitcoin price
subscription = Stream(
    streamId=StreamId(stream="bitcoin"),
    reason=StreamId(stream="bitcoin_prediction"),
    reason_is_primary=True
)

# Publish Bitcoin price prediction
publication = Stream(
    streamId=StreamId(stream="bitcoin_prediction"),
    predicting=StreamId(stream="bitcoin")
)
```

### Current Implementation (Central)

**Separate predictions table:**
- Predictions stored separately from observations
- Links to streams via `stream_name` foreign key
- Not part of stream metadata itself

### Nostr Implementation

**Tag-based prediction categorization:**
- No special prediction fields in metadata
- Use tags to mark prediction streams:

```python
# Raw data stream
DatastreamMetadata(
    stream_name="btc-price",
    tags=["bitcoin", "price", "raw_data"]
)

# Prediction stream
DatastreamMetadata(
    stream_name="btc-price-forecast",
    tags=["bitcoin", "prediction", "predicts:btc-price", "model:lstm"]
)

# Meta-prediction
DatastreamMetadata(
    stream_name="forecast-accuracy",
    tags=["meta_prediction", "predicts:btc-price-forecast"]
)
```

**Discovery:**
```python
# Find all predictions of Bitcoin price
predictions = await client.discover_datastreams(
    tags=["predicts:btc-price"]
)

# Find all meta-predictions
meta = await client.discover_datastreams(
    tags=["meta_prediction"]
)
```

**Differences:**
- Simpler (just tags, not special fields)
- More flexible (unlimited prediction chains)
- Self-documenting (tags show relationships)
- No complex `StreamPair` matching needed

---

## 8. ENCRYPTION

### Current Implementation (Neuron)

**No encryption:**
- Observations sent in plaintext over Centrifugo
- Would need separate encryption layer
- Security depends on transport (WSS)

### Current Implementation (Central)

**No encryption:**
- HTTP/HTTPS transport security only
- Data stored in database unencrypted
- Anyone with API access can read observations

### Nostr Implementation

**Built-in NIP-04 encryption:**
- `encrypted: bool` field in metadata
- Paid streams automatically encrypted
- Each observation encrypted per subscriber
- Only recipient can decrypt

**Implementation:**
```python
# Provider publishes
metadata = DatastreamMetadata(encrypted=True)
obs = DatastreamObservation(value={"price": 45000})
await client.publish_observation(obs, metadata)
# → Encrypted separately for each subscriber

# Subscriber receives
async for inbound in client.observations():
    obs = inbound.observation  # Already decrypted!
    print(obs.value)  # {"price": 45000}
```

**Advantages:**
- Privacy by default for paid streams
- Standard Nostr encryption
- No additional infrastructure needed

---

## 9. DATA FORMAT

### Current Implementation (Neuron)

**Observation structure:**
```python
Observation(
    streamId=StreamId(...),
    observationTime="2024-04-13 17:53:00.661619",
    observationHash="abc",
    value=4.2,  # Can be any type
    df=pd.DataFrame(...)  # Pandas DataFrame
)
```

**Characteristics:**
- Includes pandas DataFrame
- Observation hash for integrity
- Flexible value type
- Multi-column support via DataFrame

### Current Implementation (Central)

**Database structure:**
```sql
observations (
    id INTEGER,
    stream_name INTEGER,
    value TEXT,
    observed_at TEXT,
    hash TEXT,
    ts TIMESTAMP
)
```

**Characteristics:**
- Simple flat structure
- Value stored as text
- No typing enforcement

### Nostr Implementation

**Observation structure:**
```python
DatastreamObservation(
    stream_name="btc-price",
    timestamp=1234567890,  # Unix timestamp
    value={"price": 45000, "volume": 100},  # Any JSON-serializable
    seq_num=42  # Sequence number
)
```

**Characteristics:**
- Clean, simple structure
- Sequence numbers for ordering/payments
- JSON-serializable values (dict, list, number, string)
- No hash field (Nostr events have built-in IDs)

**Differences:**
- No pandas integration (simpler)
- Sequence numbers added (for payment tracking)
- Unix timestamps (standard)
- No observation hash (Nostr event IDs serve this purpose)

---

## 10. SUMMARY COMPARISON TABLE

| Feature | Current Neuron | Current Central | Nostr Implementation |
|---------|----------------|-----------------|---------------------|
| **Identifier** | 4-part hierarchy | Simple name + UUID | stream_name + pubkey |
| **Categorization** | Fixed hierarchy | Single name field | Flexible tags |
| **Discovery** | Topic matching | Name query | Tag-based search |
| **Pricing** | Field exists | No support | Built-in payment flow |
| **Payments** | Not implemented | No support | Encrypted notifications |
| **Delivery** | Centrifugo pubsub | HTTP POST | Nostr relays (DM) |
| **Encryption** | No | No | Yes (NIP-04) |
| **Health Tracking** | Cadence field | No | Cadence + relay queries |
| **Predictions** | Special fields | Separate table | Tags ("predicts:") |
| **Infrastructure** | Centralized broker | Central server | Decentralized relays |
| **Access Control** | No | No | Payment-based |
| **Accountability** | No | No | Public subscriptions |
| **Dependencies** | Centrifugo server | PostgreSQL + API | Nostr relays only |
| **Complexity** | High (4-part ID) | Medium | Low (simple ID) |
| **Flexibility** | Low (fixed structure) | Low | High (tag-based) |

---

## 11. FIELDS MISSING FROM NOSTR

### From Neuron Stream

**Unused fields (can ignore):**
- `offset` - UTC offset (unused)
- `datatype` - Data type (unused)
- `url` - Source URL (unused)
- `hook` - Webhook endpoint (unused)
- `history` - Historical data info (unused)
- `pinned` - Is stream pinned (UI concern, not protocol)

**Potentially useful fields:**
- ❓ `uri` - Could be useful for identifying data sources
- ❓ `headers` / `payload` - For API-based streams

**Prediction fields:**
- ❌ `predicting: StreamId` - Replaced with tags: `["predicts:stream-id"]`
- ❌ `reason: StreamId` - Not needed (tags handle this)
- ❌ `reason_is_primary` - Not needed

### From Central Database

**All fields covered:**
- `name` → `name`
- `author` → `nostr_pubkey`
- `secondary` → Can use tags or description
- `target` → Can encode in stream_name or use tags
- `meta` → Can use tags or description
- `description` → `description`

**Database-specific:**
- `id` (auto-increment) - Not needed (stream_name is key)
- `uuid` - Generated by Nostr event system
- `ts` (creation time) - Have `created_at`

---

## 12. RECOMMENDED MAPPING

### Converting Current → Nostr

```python
# Current Neuron Stream
current = Stream(
    streamId=StreamId(
        source="Streamr",
        author="02a85fb71485...",
        stream="DATAUSD/binance/ticker",
        target="Close"
    ),
    cadence=3600,
    price_per_obs=10,
    description="Bitcoin price from Binance"
)

# Maps to Nostr DatastreamMetadata
nostr = DatastreamMetadata(
    stream_name="DATAUSD-binance-ticker-Close",  # Flatten hierarchy
    nostr_pubkey="02a85fb71485...",
    name="DATAUSD Binance Ticker (Close)",
    description="Bitcoin price from Binance",
    encrypted=True,  # If price > 0
    price_per_obs=10,
    created_at=int(time.time()),
    cadence_seconds=3600,
    tags=[
        "Streamr",  # Original source
        "DATAUSD",
        "binance",
        "ticker",
        "Close",
        "price",
        "raw_data"
    ]
)
```

### Converting Central → Nostr

```python
# Current Central Stream
central = Stream(
    id=123,
    uuid="550e8400-e29b-41d4-a716-446655440000",
    name="bitcoin",
    author="02a85fb71485...",
    secondary="binance",
    target="price",
    description="Bitcoin price observations"
)

# Maps to Nostr DatastreamMetadata
nostr = DatastreamMetadata(
    stream_name="bitcoin",  # Or "bitcoin-binance-price" if more specific needed
    nostr_pubkey="02a85fb71485...",
    name="Bitcoin",
    description="Bitcoin price observations",
    encrypted=False,  # Default, update based on use case
    price_per_obs=0,  # Default, update based on use case
    created_at=int(central.ts.timestamp()),
    cadence_seconds=3600,  # Need to determine this
    tags=[
        "bitcoin",
        "binance",  # From secondary
        "price",    # From target
        "raw_data"
    ]
)
```

---

## 13. KEY DIFFERENCES & DESIGN PHILOSOPHY

### Current Implementation Philosophy

**Neuron:**
- Complex hierarchical structure
- Mixed concerns (data source config + metadata + predictions)
- Pandas-centric
- Tightly coupled to Centrifugo
- Many unused fields ("for future use")

**Central:**
- Simple database storage
- Minimal metadata
- HTTP/REST focused
- Separate tables for different concerns

### Nostr Implementation Philosophy

**Design principles:**
- **Simple** - Flat structure, no hierarchy
- **Flexible** - Tags for categorization
- **Focused** - Every field actively used
- **Decentralized** - Works with any relay
- **Self-contained** - No external dependencies beyond Nostr
- **Payment-native** - Micropayments built in
- **Privacy-aware** - Encryption by default for paid streams

**Tradeoffs:**
- ✅ Simpler to understand and use
- ✅ More flexible categorization (tags vs hierarchy)
- ✅ Built-in payments and encryption
- ✅ Decentralized (no single point of failure)
- ❌ No pandas integration (simpler data model)
- ❌ No API source config fields (streams are outputs, not inputs)

---

## 14. MIGRATION RECOMMENDATIONS

### What to Keep from Current

1. **Cadence concept** - ✅ Already in Nostr
2. **Price per observation** - ✅ Already in Nostr
3. **Predictions as streams** - ✅ Handle via tags

### What to Drop

1. **4-part hierarchy** - Replace with flat ID + tags
2. **Unused fields** - Don't migrate
3. **Pandas DataFrames** - Use simple JSON values
4. **Observation hashes** - Use Nostr event IDs
5. **Complex source config** - Streams publish data, don't fetch it

### What to Add

1. **Payment notifications** - New protocol
2. **Encryption** - Built into Nostr
3. **Public subscriptions** - Accountability
4. **Tag-based discovery** - More flexible
5. **Stream health checking** - Query relay for observation times

### Migration Path

```
1. Create tag mapping from hierarchical IDs
   source/author/stream/target → flat stream_name + tags

2. Set appropriate cadence_seconds based on historical data

3. Determine pricing (price_per_obs) per stream

4. Mark prediction streams with "predicts:" tags

5. Publish announcements to Nostr relays

6. Migrate observations to new format (remove DataFrame, add seq_num)

7. Update subscribers to use new discovery API

8. Deploy payment infrastructure (if using paid streams)
```

---

## 15. QUESTIONS TO RESOLVE

### 1. API Source Streams

**Current:** Neuron streams have `url`, `headers`, `payload` for fetching from APIs

**Nostr:** These fields don't exist - streams are outputs only

**Question:** Do we need to track where data comes from, or is that internal to the provider?

**Recommendation:**
- Keep data source config separate (provider-side only)
- Streams represent outputs, not inputs
- Provider can mention source in description or tags

### 2. Target Field

**Current:** Target specifies which field in structured data (e.g., "Close" price)

**Nostr:** No target field

**Options:**
1. Encode in stream_name: `"btc-binance-close"`
2. Use tags: `["target:Close"]`
3. Include all fields in observation value, let subscriber extract

**Recommendation:** Option 3 - include all fields:
```python
value = {
    "open": 44000,
    "high": 45500,
    "low": 43500,
    "close": 45000,  # Subscriber extracts what they need
    "volume": 1000
}
```

### 3. Database Storage

**Current:** Central stores observations in PostgreSQL

**Nostr:** Relays store events, but providers might want local DB

**Question:** Should providers maintain a local database of observations?

**Recommendation:**
- Providers can optionally cache locally
- Primary source of truth is Nostr relays
- Use relays for queries when possible
- Local DB only for performance/analytics

### 4. Observation Hashes

**Current:** Each observation has a hash for integrity

**Nostr:** Events have IDs but no separate content hash

**Question:** Do we need observation-level content hashes?

**Recommendation:**
- Nostr event IDs provide integrity (signed by provider)
- Additional hashing not needed
- Can add to observation value if required: `value = {"price": 45000, "hash": "abc"}`

---

## CONCLUSION

The Nostr implementation is **simpler, more focused, and more flexible** than the current implementations:

✅ **Simpler identification** - Flat stream_name instead of 4-part hierarchy
✅ **Better discovery** - Flexible tags instead of rigid structure
✅ **Built-in payments** - Protocol-level micropayments
✅ **Privacy** - Encryption by default for paid streams
✅ **Decentralized** - No dependency on central infrastructure
✅ **Health tracking** - Standard way to check stream activity
✅ **Accountability** - Public subscriptions and metadata

The main tradeoff is **reduced data source configuration** (no url/headers/payload fields), but this is actually cleaner - streams represent outputs, not inputs.

**Recommendation:** Adopt Nostr implementation and phase out current approaches. The benefits far outweigh the migration effort.
