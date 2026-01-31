---
sidebar_position: 13
title: Streaming
---

# Streaming Generation

Generate synthetic data in real-time streams for continuous data pipelines.

## Quick Start

```python
from genesis.generators.streaming import StreamingGenerator

# Load pre-trained generator
generator = SyntheticGenerator.load('trained_model.pkl')

# Create streaming wrapper
streamer = StreamingGenerator(generator)

# Stream batches
for batch in streamer.stream(batch_size=100, interval_seconds=1.0):
    process_batch(batch)
```

## Streaming Modes

### Pull Mode (Iterator)

Consume data at your own pace:

```python
streamer = StreamingGenerator(generator)

for batch in streamer.stream(batch_size=100):
    # Process each batch
    save_to_database(batch)
    
    # Stop when you have enough
    if total_records >= 10000:
        break
```

### Push Mode (Publisher)

Push to external systems:

```python
# Push to Kafka
from genesis.generators.streaming import KafkaPublisher

publisher = KafkaPublisher(
    bootstrap_servers='localhost:9092',
    topic='synthetic-data'
)

streamer.stream_to(
    publisher,
    batch_size=100,
    interval_seconds=0.5,
    max_batches=1000
)
```

## Kafka Integration

### Setup

```python
from genesis.generators.streaming import StreamingGenerator, KafkaPublisher

# Load generator
generator = SyntheticGenerator.load('model.pkl')
streamer = StreamingGenerator(generator)

# Configure Kafka publisher
publisher = KafkaPublisher(
    bootstrap_servers='localhost:9092',
    topic='synthetic-customers',
    value_serializer='json',  # 'json' or 'avro'
    acks='all',
    compression='gzip'
)

# Start streaming
streamer.stream_to(
    publisher,
    batch_size=100,
    interval_seconds=1.0
)
```

### With Avro Schema

```python
schema = {
    "type": "record",
    "name": "Customer",
    "fields": [
        {"name": "id", "type": "string"},
        {"name": "age", "type": "int"},
        {"name": "income", "type": "float"}
    ]
}

publisher = KafkaPublisher(
    bootstrap_servers='localhost:9092',
    topic='customers',
    value_serializer='avro',
    avro_schema=schema
)
```

## WebSocket Integration

```python
from genesis.generators.streaming import StreamingGenerator, WebSocketPublisher

publisher = WebSocketPublisher(
    uri='ws://localhost:8765',
    format='json'
)

# Async streaming
async def stream_data():
    await streamer.stream_to_async(
        publisher,
        batch_size=50,
        interval_seconds=0.1
    )

asyncio.run(stream_data())
```

### WebSocket Server

```python
import asyncio
import websockets
from genesis.generators.streaming import StreamingGenerator

async def ws_handler(websocket, path):
    """Handle WebSocket connections."""
    streamer = StreamingGenerator(generator)
    
    async for batch in streamer.stream_async(batch_size=10):
        await websocket.send(batch.to_json())
        await asyncio.sleep(0.1)

# Start server
start_server = websockets.serve(ws_handler, "localhost", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

## Conditional Streaming

Stream data matching conditions:

```python
# Stream only high-value customers
for batch in streamer.stream(
    batch_size=100,
    conditions={'segment': 'premium', 'lifetime_value': ('>', 10000)}
):
    process_premium_customers(batch)
```

## Rate Limiting

Control generation rate:

```python
# Fixed rate
streamer.stream_to(
    publisher,
    rate_limit=1000,  # 1000 records per second
)

# Adaptive rate
streamer.stream_to(
    publisher,
    adaptive_rate=True,
    target_latency_ms=100
)
```

## Backpressure Handling

Handle slow consumers:

```python
streamer = StreamingGenerator(
    generator,
    buffer_size=10000,          # Buffer up to 10K records
    on_buffer_full='drop',      # 'drop', 'block', or 'error'
    drop_oldest=True
)
```

## Monitoring

```python
from genesis.generators.streaming import StreamingGenerator, StreamingMetrics

streamer = StreamingGenerator(generator)
metrics = StreamingMetrics()

# Start streaming with metrics
streamer.stream_to(
    publisher,
    batch_size=100,
    metrics_collector=metrics
)

# Check metrics
print(f"Total records: {metrics.total_records}")
print(f"Records/second: {metrics.records_per_second}")
print(f"Batches sent: {metrics.batches_sent}")
print(f"Errors: {metrics.errors}")
```

## Error Handling

```python
from genesis.generators.streaming import StreamingGenerator, RetryPolicy

retry_policy = RetryPolicy(
    max_retries=3,
    backoff_factor=2,
    max_backoff_seconds=60
)

streamer = StreamingGenerator(
    generator,
    retry_policy=retry_policy,
    on_error='retry'  # 'retry', 'skip', 'stop'
)

try:
    streamer.stream_to(publisher, batch_size=100)
except StreamingError as e:
    logger.error(f"Streaming failed: {e}")
```

## Complete Example

```python
import logging
from genesis import SyntheticGenerator
from genesis.generators.streaming import (
    StreamingGenerator, 
    KafkaPublisher,
    StreamingMetrics
)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load trained generator
generator = SyntheticGenerator.load('customer_generator.pkl')

# Create streaming wrapper
streamer = StreamingGenerator(
    generator,
    buffer_size=5000,
    on_buffer_full='block'
)

# Configure Kafka
publisher = KafkaPublisher(
    bootstrap_servers='kafka:9092',
    topic='synthetic-customers',
    value_serializer='json',
    compression='gzip'
)

# Metrics collection
metrics = StreamingMetrics()

# Start streaming
try:
    streamer.stream_to(
        publisher,
        batch_size=100,
        interval_seconds=0.5,
        conditions={'status': 'active'},
        metrics_collector=metrics,
        max_records=100000
    )
except KeyboardInterrupt:
    print("Streaming stopped")
finally:
    print(f"Total records sent: {metrics.total_records}")
    print(f"Average rate: {metrics.average_rate:.1f} records/sec")
```

## CLI Usage

```bash
# Start streaming to stdout
genesis stream model.pkl --batch-size 100 --interval 1

# Stream to Kafka
genesis stream model.pkl \
  --kafka-servers localhost:9092 \
  --topic synthetic-data \
  --batch-size 100 \
  --rate-limit 1000

# Stream with conditions
genesis stream model.pkl \
  --conditions '{"status": "active"}' \
  --max-records 10000
```

## Best Practices

1. **Set appropriate batch sizes** - Balance throughput and latency
2. **Monitor metrics** - Track rate and errors
3. **Handle backpressure** - Configure buffer and overflow behavior
4. **Use compression** - Reduce network bandwidth
5. **Test failure scenarios** - Ensure retry logic works

## Next Steps

- **[Conditional Generation](/docs/guides/conditional-generation)** - Filter streaming data
- **[Distributed Generation](/docs/advanced/distributed)** - Scale streaming
- **[Pipelines](/docs/guides/pipelines)** - Integrate with workflows
