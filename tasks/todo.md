# Engine-Neuron Integration Plan

## Progress Summary - ALL PHASES COMPLETE

### Completed:
- [x] Phase 1.1: Commented out centrifugo code in Neuron
- [x] Phase 1.2: Commented out data-related code in Neuron
- [x] Phase 2.1: Commented out centrifugo code in Engine
- [x] Phase 2.2: Commented out P2P code in Engine
- [x] Phase 3: Implemented Neuron spawning Engine
- [x] Phase 4.1: Updated Dockerfile to include engine-lite
- [x] Phase 5.1: Engine fetches data from Central Server
- [x] Phase 5.3: Engine publishes predictions to Central Server

### Remaining (Future Work):
- [ ] Phase 5.2: Implement SQLite storage in Engine (can reuse DataManager patterns)

---

## Review Summary

### Changes Made:

#### 1. Neuron (`neuron-lite/start.py`)
- **Import added**: `from satoriengine.veda.engine import Engine`
- **Attribute added**: `self.aiengine: Union[Engine, None] = None`
- **Method added**: `spawnEngine()` - spawns Engine with stream assignments
- **start() updated**: Calls `self.spawnEngine()` after `getBalances()`
- **Commented out**:
  - Centrifugo imports, attributes, and methods
  - `connectToDataServer()`, `stayConnectedForever()`
  - `dataServerFinalize()`, `sharePubSubInfo()`, `populateData()`
  - `subscribeToRawData()`, `subscribeToEngineUpdates()`
  - `handleRawData()`, `handlePredictionData()`
  - `publish()` method

#### 2. Engine (`engine-lite/engine.py`)
- **Import added**: `from satorilib.server import SatoriServerClient`
- **New factory method**: `Engine.createFromNeuron()` - creates Engine from Neuron's stream assignments
- **New attributes**: `server`, `wallet`, `subscriptionStreams`, `publicationStreams`, `useServer`
- **New method**: `initializeFromNeuron()` - initializes without DataServer
- **New method**: `initializeModelsFromNeuron()` - creates models using server directly
- **StreamModel new factory**: `createFromServer()` - creates model using Central Server
- **StreamModel new methods**:
  - `initializeFromServer()` - initializes model in server mode
  - `loadDataFromServer()` - fetches data from Central Server
  - `publishPredictionToServer()` - publishes predictions to Central Server
- **Updated**: `passPredictionData()` - routes to server when `useServer=True`
- **Commented out**:
  - Centrifugo imports and methods
  - P2P methods: `p2pInit()`, `connectToPeer()`, `monitorPublisherConnection()`, `syncData()`, `makeSubscription()`, etc.

#### 3. Package Structure (`engine-lite/satoriengine/veda/`)
- Created proper namespace package structure
- `config.py` - basic config for engine paths
- Symlinks to adapters, data, engine.py for import compatibility

#### 4. Dockerfile
- Added: `COPY engine-lite /Satori/Engine`

---

## Architecture After Changes

```
┌─────────────────────────────────────────────────────────────┐
│                         Neuron                               │
│  - Checks in with Central Server                            │
│  - Gets stream assignments (subscriptions/publications)      │
│  - Wallet auth                                               │
│  - Spawns Engine with stream assignments                     │
└──────────────────────────┬──────────────────────────────────┘
                           │ createFromNeuron()
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                         Engine                               │
│  - Receives stream assignments from Neuron                   │
│  - Fetches historical data from Central Server               │
│  - Trains models on data                                     │
│  - Makes predictions                                         │
│  - Publishes predictions directly to Central Server          │
│  - Stores data/models locally                                │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Neuron starts** → checks in with Central Server → gets streams
2. **Neuron spawns Engine** → passes subscriptions, publications, server client
3. **Engine initializes** → fetches historical data from Central Server
4. **Engine trains models** → runs prediction loop
5. **Engine publishes predictions** → directly to Central Server

## Files Modified

| File | Changes |
|------|---------|
| `neuron-lite/start.py` | Added Engine spawn, commented out centrifugo/data code |
| `engine-lite/engine.py` | Added server mode, commented out centrifugo/P2P code |
| `engine-lite/__init__.py` | Created package init |
| `engine-lite/satoriengine/__init__.py` | Created namespace package |
| `engine-lite/satoriengine/veda/__init__.py` | Created namespace package |
| `engine-lite/satoriengine/veda/config.py` | Created config module |
| `Dockerfile` | Added engine-lite copy |

## Next Steps (Optional)

1. **SQLite Storage**: Add local SQLite database for caching stream data and predictions
2. **Testing**: Test the integration end-to-end
3. **Error Handling**: Add more robust error handling for server communication failures
4. **Reconnection Logic**: Add logic to handle Central Server disconnections
