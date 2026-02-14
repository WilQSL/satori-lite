# Metadata Field Implementation Summary

## What Was Implemented

Added an optional `metadata` field to `DatastreamMetadata` to provide an extension point for domain-specific information without bloating the core protocol.

---

## Changes Made

### 1. Model Update (`models.py`)

**Added field:**
```python
@dataclass
class DatastreamMetadata:
    # ... existing fields ...
    metadata: dict[str, Any] | None = None  # Optional: source info, lineage, model details, etc.
```

**Characteristics:**
- Optional (defaults to `None`)
- Can contain any JSON-serializable dictionary
- Backward compatible (existing code works without changes)

### 2. Tests Added (`test_satori_nostr_models.py`)

Added 4 new comprehensive tests:

1. **`test_metadata_field_optional`** - Verifies field defaults to None
2. **`test_metadata_field_with_source_info`** - Tests source information storage
3. **`test_metadata_field_with_lineage`** - Tests data lineage and model info
4. **`test_metadata_json_serialization`** - Tests JSON round-trip with metadata

**Test results:** 25/25 passing ✅

### 3. Documentation

Created two comprehensive documentation files:

**`METADATA_EXAMPLES.md`** - Extensive examples covering:
- API source tracking (REST, WebSocket)
- Data lineage and provenance
- ML model documentation
- IoT/sensor data
- Migration from legacy systems
- Quality metrics and SLAs
- Versioning strategies
- Best practices

**`METADATA_IMPLEMENTATION.md`** - This file

---

## Usage Examples

### Simple Stream (No Metadata)

```python
from satori_nostr import DatastreamMetadata, CADENCE_HOURLY
import time

metadata = DatastreamMetadata(
    stream_name="btc-price",
    nostr_pubkey="abc123...",
    name="Bitcoin Price",
    description="BTC/USD spot price",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=CADENCE_HOURLY,
    tags=["bitcoin", "price"]
    # metadata not provided - defaults to None
)
```

### With API Source Information

```python
metadata = DatastreamMetadata(
    stream_name="btc-coinbase",
    nostr_pubkey="abc123...",
    name="Bitcoin Coinbase",
    description="BTC from Coinbase API",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=60,
    tags=["bitcoin", "coinbase", "api"],
    metadata={
        "version": "1.0",
        "source": {
            "type": "rest_api",
            "url": "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            "method": "GET",
            "provider": "Coinbase"
        },
        "target": "price"  # Which field to extract
    }
)
```

### With Data Lineage

```python
metadata = DatastreamMetadata(
    stream_name="btc-cleaned",
    # ... core fields ...
    metadata={
        "version": "1.0",
        "lineage": {
            "original_source": "Coinbase API",
            "source_stream_name": "btc-coinbase",
            "transformations": [
                {
                    "step": 1,
                    "type": "outlier_removal",
                    "method": "IQR"
                },
                {
                    "step": 2,
                    "type": "smoothing",
                    "method": "moving_average",
                    "window": 10
                }
            ],
            "quality_score": 0.95
        }
    }
)
```

### With ML Model Information

```python
metadata = DatastreamMetadata(
    stream_name="btc-lstm-forecast",
    # ... core fields ...
    tags=["bitcoin", "prediction", "predicts:btc-price", "model:lstm"],
    metadata={
        "version": "1.0",
        "model": {
            "type": "LSTM",
            "architecture": "2-layer-128-units",
            "framework": "tensorflow",
            "performance": {
                "accuracy": 0.85,
                "mse": 0.0023,
                "r2_score": 0.85
            }
        },
        "prediction": {
            "horizon": "24h",
            "target_stream": "btc-price"
        },
        "lineage": {
            "input_streams": ["btc-price", "btc-volume", "btc-sentiment"]
        }
    }
)
```

---

## Design Benefits

### 1. **Separation of Concerns**
- **Core protocol** = Simple, well-defined fields
- **Metadata** = Domain-specific extensions

### 2. **Backward Compatible**
- Existing streams work without changes
- Field is optional (defaults to None)
- No breaking changes

### 3. **Flexible & Extensible**
- Add any custom fields
- Version your metadata schema
- Evolve without protocol changes

### 4. **Preserves Information**
- Track data provenance
- Document transformations
- Credit original sources
- Store quality metrics

### 5. **Independently Versioned**
```python
# Version 1.0
metadata = {
    "version": "1.0",
    "source": {...}
}

# Future: Version 2.0 with new fields
metadata = {
    "version": "2.0",
    "source": {...},
    "compliance": {...},  # New in v2.0
    "licensing": {...}    # New in v2.0
}
```

---

## Migration from Current System

### Current Hierarchical Stream

```python
# Old Neuron format
StreamId(
    source="Streamr",
    author="pubkey",
    stream="DATAUSD/binance/ticker",
    target="Close"
)
```

### Maps to Nostr with Metadata

```python
DatastreamMetadata(
    stream_name="DATAUSD-binance-ticker",
    nostr_pubkey="pubkey",
    name="DATAUSD Binance Ticker",
    description="Bitcoin ticker from Binance",
    # ... standard fields ...
    tags=["DATAUSD", "binance", "ticker", "Close"],
    metadata={
        "version": "1.0",
        "legacy": {
            "source": "Streamr",
            "stream": "DATAUSD/binance/ticker",
            "target": "Close"
        },
        "source": {
            "type": "rest_api",
            "url": "https://api.binance.com/...",
            "target_field": "Close"
        }
    }
)
```

---

## Standard Metadata Schema (Recommended)

### Version 1.0

```python
{
    "version": "1.0",  # Required: schema version

    # Optional sections (include as needed):

    "source": {
        "type": "rest_api | websocket | iot_sensor | manual",
        "url": "...",
        "provider": "..."
    },

    "lineage": {
        "original_source": "...",
        "source_stream_name": "...",
        "transformations": [...]
    },

    "model": {
        "type": "LSTM | RandomForest | ...",
        "framework": "...",
        "performance": {...}
    },

    "prediction": {
        "horizon": "...",
        "target_stream": "..."
    },

    "quality": {
        "quality_score": 0.95,
        "data_completeness": 0.99
    },

    "legacy": {
        # Preserve old system identifiers
    }
}
```

---

## Reading Metadata in Code

### Safe Access Pattern

```python
def get_metadata_field(stream: DatastreamMetadata, *keys, default=None):
    """Safely navigate nested metadata."""
    if not stream.metadata:
        return default

    value = stream.metadata
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

# Usage
source_url = get_metadata_field(stream, "source", "url", default="unknown")
model_type = get_metadata_field(stream, "model", "type", default="unknown")
```

### Check for Specific Metadata

```python
# Check if stream has source info
if stream.metadata and "source" in stream.metadata:
    source_type = stream.metadata["source"]["type"]
    print(f"Data from {source_type}")

# Check if stream is a prediction
if stream.metadata and "model" in stream.metadata:
    accuracy = stream.metadata["model"]["performance"]["accuracy"]
    print(f"Prediction accuracy: {accuracy}")
```

---

## Best Practices

### 1. Always Version Your Metadata

```python
metadata = {
    "version": "1.0",  # Always include
    # ... rest
}
```

### 2. Keep It Optional

Don't require metadata for simple streams:
```python
# Simple stream - no metadata
simple = DatastreamMetadata(...)

# Complex stream - metadata when useful
complex = DatastreamMetadata(..., metadata={...})
```

### 3. Use Consistent Structure

Group related information:
```python
{
    "version": "1.0",
    "source": {...},    # Source-related
    "lineage": {...},   # Lineage-related
    "model": {...},     # Model-related
    "quality": {...}    # Quality-related
}
```

### 4. Document Your Schema

If creating custom metadata:
```python
{
    "version": "1.0",
    "schema": "acme-corp-standard-v1",
    "schema_url": "https://docs.acme.com/stream-metadata-v1",
    # ... your fields
}
```

### 5. JSON-Serializable Only

Metadata must be JSON-serializable:
- ✅ dict, list, str, int, float, bool, None
- ❌ functions, classes, file handles

---

## Testing

All tests passing: **25/25** ✅

New tests added:
- Metadata field optional
- Metadata with source info
- Metadata with lineage
- JSON serialization round-trip

Backward compatibility verified:
- Existing streams work without metadata
- All integration tests pass

---

## Files Modified

1. **`satori_nostr/models.py`**
   - Added `metadata` field to `DatastreamMetadata`

2. **`tests/test_satori_nostr_models.py`**
   - Added 4 new tests for metadata field

3. **`satori_nostr/METADATA_EXAMPLES.md`** (new)
   - Comprehensive usage examples

4. **`satori_nostr/METADATA_IMPLEMENTATION.md`** (new)
   - This summary document

---

## Summary

The metadata field provides a **clean extension point** for:

✅ Tracking data sources (API endpoints, sensors, etc.)
✅ Preserving data lineage and provenance
✅ Documenting ML models and predictions
✅ Storing quality metrics
✅ Migrating from legacy systems
✅ Future extensibility

**Key principle:** Core protocol stays simple and focused, metadata handles complex/domain-specific cases.

The implementation is:
- ✅ **Optional** - Only use when needed
- ✅ **Flexible** - Store any JSON-serializable data
- ✅ **Versioned** - Evolve schema over time
- ✅ **Backward compatible** - No breaking changes
- ✅ **Well-tested** - All tests passing
- ✅ **Documented** - Extensive examples provided
