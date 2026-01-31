# ADR-0014: Streaming Generation with Kafka/WebSocket Integration

## Status

Accepted

## Context

Traditional synthetic data generation follows a batch paradigm: fit a model, generate N samples, write to file. However, modern data architectures require:

1. **Real-time pipelines**: Kafka/Kinesis streams need continuous synthetic data
2. **Memory efficiency**: Generating 100M rows at once is impractical
3. **Incremental updates**: Models should adapt to new data without full retraining
4. **Interactive applications**: WebSocket APIs for dashboards and demos

We needed a streaming architecture that:

- Generates data incrementally (iterator/generator pattern)
- Integrates with enterprise message brokers
- Supports partial model updates
- Provides backpressure handling

## Decision

We implement a **streaming generation layer** with multiple integration points:

```python
from genesis.streaming import StreamingGenerator, KafkaStreamingGenerator

# Basic streaming (Python generator pattern)
stream_gen = StreamingGenerator(method='gaussian_copula')
stream_gen.fit(initial_data)

for batch in stream_gen.generate_stream(n_batches=100, batch_size=1000):
    process_batch(batch)  # DataFrame of 1000 rows

# Kafka integration
kafka_gen = KafkaStreamingGenerator(
    bootstrap_servers="localhost:9092",
    input_topic="real-data",
    output_topic="synthetic-data",
)
kafka_gen.fit_from_topic(timeout_seconds=60)
kafka_gen.start_streaming(generate_ratio=1.0)

# WebSocket server
from genesis.streaming import WebSocketStreamingGenerator

ws_gen = WebSocketStreamingGenerator(generator, host="0.0.0.0", port=8765)
ws_gen.run()  # Serves synthetic data over WebSocket
```

Key components:

1. **`StreamingGenerator`**: Wraps any generator with streaming capabilities
2. **`KafkaStreamingGenerator`**: Consumes from Kafka, produces synthetic data back
3. **`WebSocketStreamingGenerator`**: Real-time API for web applications
4. **`DataStreamProcessor`**: Sliding window updates for online learning

## Consequences

### Positive

- **Memory efficient**: Only one batch in memory at a time
- **Backpressure aware**: Queue-based architecture prevents overwhelming consumers
- **Incremental learning**: `partial_fit()` updates model with new data
- **Enterprise ready**: Native Kafka integration for data platforms
- **Interactive**: WebSocket enables real-time dashboards and demos

### Negative

- **Stateful complexity**: Streaming generators maintain more state than batch
- **Consistency challenges**: Incremental updates may cause distribution drift
- **Dependency weight**: Kafka/WebSocket integrations require optional dependencies
- **Testing difficulty**: Streaming systems are harder to test deterministically

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Kafka Topic   │────▶│ KafkaStreaming   │────▶│   Kafka Topic   │
│  (real-data)    │     │    Generator     │     │ (synthetic-data)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │ StreamingGenerator│
                        │  - fit()         │
                        │  - partial_fit() │
                        │  - generate()    │
                        └──────────────────┘
                               │
                               ▼
┌─────────────────┐     ┌──────────────────┐
│   WebSocket     │◀────│ WebSocketServer  │
│   Clients       │     │                  │
└─────────────────┘     └──────────────────┘
```

### Incremental Updates

The streaming layer supports online learning through weighted updates:

```python
def partial_fit(self, new_data: pd.DataFrame, weight: float = 0.1):
    """Update model incrementally with new data.
    
    Args:
        new_data: New observations to incorporate
        weight: Influence of new data (0-1, higher = more influence)
    """
    # Combine old and new data with weighting
    combined = self._weighted_sample(
        self._original_data, 
        new_data, 
        old_weight=1-weight,
        new_weight=weight
    )
    self._generator.fit(combined)
```

## Streaming Protocols

### Kafka Message Format

```json
{
    "action": "generate",
    "n_samples": 100,
    "conditions": {"fraud": true}
}
```

### WebSocket Protocol

```json
// Request
{"action": "generate", "n_samples": 100}
{"action": "stream", "n_batches": 10, "batch_size": 100}
{"action": "stats"}

// Response
{"action": "data", "data": [...], "n_samples": 100}
{"action": "batch", "data": [...], "n_samples": 100}
{"action": "complete"}
```

## Examples

```python
# Memory-efficient large dataset generation
from genesis.streaming import generate_streaming

# Generates 10M rows in 10K-row batches
for batch in generate_streaming(data, n_samples=10_000_000, batch_size=10_000):
    batch.to_parquet(f"output/batch_{i}.parquet")

# Async generation with callback
def handle_batch(batch):
    kafka_producer.send("synthetic", batch.to_dict(orient="records"))

stream_gen.generate_async(callback=handle_batch, n_batches=1000)

# Real-time dashboard backend
@app.websocket("/synthetic")
async def synthetic_endpoint(websocket):
    ws_gen = WebSocketStreamingGenerator(fitted_generator)
    await ws_gen.handler(websocket, "/synthetic")
```
