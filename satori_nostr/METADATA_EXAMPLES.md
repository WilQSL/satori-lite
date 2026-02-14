# Metadata Field Usage Examples

The optional `metadata` field in `DatastreamMetadata` provides an extension point for domain-specific information without bloating the core protocol.

---

## Basic Usage

### Stream Without Metadata (Simple Case)

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
    tags=["bitcoin", "price", "usd"]
    # metadata field omitted - defaults to None
)
```

---

## API Source Tracking

### REST API Data Source

```python
metadata = DatastreamMetadata(
    stream_name="btc-coinbase",
    nostr_pubkey="abc123...",
    name="Bitcoin Price from Coinbase",
    description="Real-time BTC/USD from Coinbase API",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=60,
    tags=["bitcoin", "price", "coinbase", "api"],
    metadata={
        "version": "1.0",
        "source": {
            "type": "rest_api",
            "provider": "Coinbase",
            "url": "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            "method": "GET",
            "headers": {
                "User-Agent": "Satori/1.0"
            },
            "response_path": "data.amount",  # JSON path to extract
            "update_frequency": 60  # seconds
        },
        "target": "price"  # Which field from response
    }
)
```

### WebSocket API Source

```python
metadata = DatastreamMetadata(
    stream_name="btc-binance-ws",
    nostr_pubkey="abc123...",
    name="Bitcoin Binance WebSocket",
    description="Real-time BTC/USD from Binance WebSocket",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=1,  # Real-time
    tags=["bitcoin", "price", "binance", "websocket", "realtime"],
    metadata={
        "version": "1.0",
        "source": {
            "type": "websocket",
            "provider": "Binance",
            "url": "wss://stream.binance.com:9443/ws/btcusdt@ticker",
            "subscription": {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@ticker"]
            },
            "message_path": "c"  # Current price field
        },
        "target": "price"
    }
)
```

---

## Data Lineage & Provenance

### With Transformations

```python
metadata = DatastreamMetadata(
    stream_name="btc-cleaned",
    nostr_pubkey="abc123...",
    name="Bitcoin Price (Cleaned)",
    description="BTC/USD with outliers removed and smoothed",
    encrypted=False,
    price_per_obs=5,
    created_at=int(time.time()),
    cadence_seconds=300,
    tags=["bitcoin", "price", "processed", "cleaned"],
    metadata={
        "version": "1.0",
        "lineage": {
            "original_source": "Coinbase API",
            "source_stream_name": "btc-coinbase",  # Reference to raw stream
            "transformations": [
                {
                    "step": 1,
                    "type": "outlier_removal",
                    "method": "IQR",
                    "threshold": 3.0
                },
                {
                    "step": 2,
                    "type": "smoothing",
                    "method": "moving_average",
                    "window": 10
                }
            ],
            "quality_metrics": {
                "outliers_removed": 23,
                "data_completeness": 0.99,
                "quality_score": 0.95
            }
        },
        "processing": {
            "last_processed": 1234567890,
            "processor_version": "2.1.0"
        }
    }
)
```

### Multi-Source Aggregation

```python
metadata = DatastreamMetadata(
    stream_name="btc-vwap",
    nostr_pubkey="abc123...",
    name="Bitcoin VWAP",
    description="Volume-weighted average price from multiple exchanges",
    encrypted=True,
    price_per_obs=10,
    created_at=int(time.time()),
    cadence_seconds=60,
    tags=["bitcoin", "price", "vwap", "aggregated"],
    metadata={
        "version": "1.0",
        "lineage": {
            "type": "aggregation",
            "method": "volume_weighted_average",
            "sources": [
                {
                    "stream_name": "btc-coinbase",
                    "weight": 0.35,
                    "reliability": 0.98
                },
                {
                    "stream_name": "btc-binance",
                    "weight": 0.40,
                    "reliability": 0.99
                },
                {
                    "stream_name": "btc-kraken",
                    "weight": 0.25,
                    "reliability": 0.97
                }
            ],
            "total_volume_24h": 1500000000,  # USD
            "exchanges_count": 3
        }
    }
)
```

---

## Machine Learning Predictions

### Simple ML Model

```python
metadata = DatastreamMetadata(
    stream_name="btc-lstm-24h",
    nostr_pubkey="abc123...",
    name="Bitcoin LSTM 24h Forecast",
    description="24-hour Bitcoin price prediction using LSTM",
    encrypted=True,
    price_per_obs=50,
    created_at=int(time.time()),
    cadence_seconds=3600,
    tags=["bitcoin", "prediction", "predicts:btc-price", "model:lstm", "horizon:24h"],
    metadata={
        "version": "1.0",
        "model": {
            "type": "LSTM",
            "architecture": "2-layer-128-units",
            "framework": "tensorflow",
            "version": "2.15.0",
            "trained_on": {
                "start_date": "2020-01-01",
                "end_date": "2024-12-31",
                "samples": 175200
            },
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "dropout": 0.2
            },
            "performance": {
                "mse": 0.0023,
                "mae": 0.041,
                "r2_score": 0.85,
                "test_accuracy": 0.83
            }
        },
        "prediction": {
            "horizon": "24h",
            "target_stream": "btc-price",
            "confidence_interval": 0.95,
            "update_frequency": 3600
        },
        "lineage": {
            "input_streams": [
                "btc-price",
                "btc-volume",
                "btc-social-sentiment"
            ]
        }
    }
)
```

### Ensemble Model

```python
metadata = DatastreamMetadata(
    stream_name="btc-ensemble",
    nostr_pubkey="abc123...",
    name="Bitcoin Ensemble Prediction",
    description="Ensemble of multiple ML models for Bitcoin price prediction",
    encrypted=True,
    price_per_obs=100,
    created_at=int(time.time()),
    cadence_seconds=1800,
    tags=["bitcoin", "prediction", "predicts:btc-price", "ensemble"],
    metadata={
        "version": "1.0",
        "model": {
            "type": "ensemble",
            "method": "weighted_average",
            "models": [
                {
                    "name": "LSTM",
                    "stream_name": "btc-lstm-24h",
                    "weight": 0.40,
                    "accuracy": 0.83
                },
                {
                    "name": "RandomForest",
                    "stream_name": "btc-rf-24h",
                    "weight": 0.35,
                    "accuracy": 0.80
                },
                {
                    "name": "XGBoost",
                    "stream_name": "btc-xgb-24h",
                    "weight": 0.25,
                    "accuracy": 0.79
                }
            ],
            "performance": {
                "ensemble_accuracy": 0.87,
                "improvement_over_best": 0.04
            }
        },
        "prediction": {
            "horizon": "24h",
            "target_stream": "btc-price",
            "confidence_interval": 0.95
        }
    }
)
```

---

## IoT / Sensor Data

### Weather Station

```python
metadata = DatastreamMetadata(
    stream_name="weather-nyc-001",
    nostr_pubkey="abc123...",
    name="NYC Weather Station #001",
    description="Temperature, humidity, pressure from rooftop sensor",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=600,  # Every 10 minutes
    tags=["weather", "nyc", "temperature", "humidity", "iot"],
    metadata={
        "version": "1.0",
        "source": {
            "type": "iot_sensor",
            "device": {
                "manufacturer": "WeatherTech",
                "model": "WS-3000",
                "serial_number": "WT-NYC-001-2024",
                "firmware_version": "3.2.1"
            },
            "location": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "elevation_meters": 15,
                "address": "Manhattan, NYC"
            },
            "sensors": [
                {
                    "type": "temperature",
                    "unit": "celsius",
                    "accuracy": 0.1,
                    "range": [-40, 80]
                },
                {
                    "type": "humidity",
                    "unit": "percent",
                    "accuracy": 2,
                    "range": [0, 100]
                },
                {
                    "type": "pressure",
                    "unit": "hPa",
                    "accuracy": 0.5,
                    "range": [950, 1050]
                }
            ]
        },
        "calibration": {
            "last_calibrated": 1704067200,
            "next_calibration_due": 1719619200,
            "calibration_certificate": "CAL-2024-001"
        }
    }
)
```

---

## Migration from Legacy Systems

### Converting Hierarchical Stream

```python
# Old Satori Neuron format:
# StreamId(source="Streamr", author="pubkey", stream="DATAUSD/binance/ticker", target="Close")

metadata = DatastreamMetadata(
    stream_name="DATAUSD-binance-ticker",
    nostr_pubkey="pubkey",
    name="DATAUSD Binance Ticker",
    description="Bitcoin price ticker from Binance via Streamr",
    encrypted=False,
    price_per_obs=0,
    created_at=int(time.time()),
    cadence_seconds=60,
    tags=["DATAUSD", "binance", "ticker", "bitcoin", "streamr"],
    metadata={
        "version": "1.0",
        "legacy": {
            "source": "Streamr",
            "stream": "DATAUSD/binance/ticker",
            "target": "Close",
            "migration_date": "2024-01-15",
            "original_uuid": "550e8400-e29b-41d4-a716-446655440000"
        },
        "source": {
            "type": "rest_api",
            "provider": "Binance",
            "url": "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            "target_field": "price"
        }
    }
)
```

---

## Quality Metrics & SLAs

### Enterprise Stream with SLA

```python
metadata = DatastreamMetadata(
    stream_name="btc-premium",
    nostr_pubkey="abc123...",
    name="Bitcoin Premium Feed",
    description="Enterprise-grade Bitcoin price with guaranteed uptime",
    encrypted=True,
    price_per_obs=1000,  # Premium pricing
    created_at=int(time.time()),
    cadence_seconds=1,  # Real-time
    tags=["bitcoin", "price", "premium", "enterprise", "sla"],
    metadata={
        "version": "1.0",
        "sla": {
            "uptime_guarantee": 0.9999,  # 99.99%
            "max_latency_ms": 100,
            "data_accuracy": 0.999,
            "support_level": "24/7",
            "penalties": {
                "downtime_credit": 0.10,  # 10% credit per hour of downtime
                "latency_sla_breach": 0.05
            }
        },
        "quality": {
            "current_uptime": 0.99997,
            "avg_latency_ms": 45,
            "last_30d_accuracy": 0.9998,
            "total_observations": 2592000,
            "errors_last_30d": 5
        },
        "support": {
            "email": "support@example.com",
            "phone": "+1-800-555-0123",
            "slack": "enterprise-support"
        }
    }
)
```

---

## Versioning Metadata

### Version 1.0 (Current)

```python
metadata = {
    "version": "1.0",
    "source": {...},
    "lineage": {...}
}
```

### Future: Version 2.0

```python
# When you need new features, increment version
metadata = {
    "version": "2.0",
    "source": {...},
    "lineage": {...},
    # New in v2.0:
    "compliance": {
        "gdpr_compliant": True,
        "data_retention_days": 90,
        "pii_fields": []
    },
    "licensing": {
        "license": "CC-BY-4.0",
        "attribution_required": True
    }
}
```

**Consumers can handle different versions:**
```python
if metadata.metadata.get("version") == "1.0":
    # Handle v1.0 format
    source = metadata.metadata.get("source")
elif metadata.metadata.get("version") == "2.0":
    # Handle v2.0 format with additional fields
    compliance = metadata.metadata.get("compliance")
```

---

## Best Practices

### 1. Always Include Version

```python
metadata = {
    "version": "1.0",  # ← Always include this
    # ... rest of metadata
}
```

### 2. Use Consistent Structure

Group related information:
```python
metadata = {
    "version": "1.0",
    "source": {...},      # All source-related info
    "lineage": {...},     # All lineage-related info
    "model": {...},       # All model-related info
    "quality": {...}      # All quality-related info
}
```

### 3. Keep It Optional

Don't require metadata for simple streams:
```python
# Simple stream - no metadata needed
simple = DatastreamMetadata(
    stream_name="btc-price",
    # ... core fields only, no metadata
)

# Complex stream - metadata when useful
complex = DatastreamMetadata(
    stream_name="btc-ensemble",
    # ... core fields
    metadata={...}  # Add metadata for complex cases
)
```

### 4. Document Your Schema

If you create custom metadata structures, document them:
```python
# Our organization's metadata schema v1.0
metadata = {
    "version": "1.0",
    "schema": "acme-corp-standard-v1",
    # ... your fields
}
```

---

## Accessing Metadata in Code

### Reading Metadata

```python
# Get metadata if it exists
if stream.metadata:
    version = stream.metadata.get("version", "unknown")

    if "source" in stream.metadata:
        source_url = stream.metadata["source"].get("url")
        print(f"Data from: {source_url}")

    if "model" in stream.metadata:
        model_type = stream.metadata["model"]["type"]
        accuracy = stream.metadata["model"]["performance"]["accuracy"]
        print(f"Model: {model_type}, Accuracy: {accuracy}")
```

### Safe Navigation

```python
def get_nested_value(metadata, *keys, default=None):
    """Safely get nested value from metadata."""
    if not metadata:
        return default

    value = metadata
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

# Usage
accuracy = get_nested_value(
    stream.metadata,
    "model", "performance", "accuracy",
    default=0.0
)
```

---

## Summary

The `metadata` field provides:
- ✅ **Optional extension point** - Use only when needed
- ✅ **Source tracking** - Know where data comes from
- ✅ **Data lineage** - Document transformations
- ✅ **Model documentation** - ML model details
- ✅ **Quality metrics** - Track data quality
- ✅ **Versioned** - Evolve schema over time
- ✅ **Backward compatible** - Old streams work fine

**Key principle:** Core protocol stays simple, metadata handles complex/domain-specific cases.
