# Streaming Generation

Genesis supports streaming synthetic data generation for real-time pipelines, continuous data feeds, and memory-efficient processing.

## Overview

| Class | Backend | Use Case |
|-------|---------|----------|
| **StreamingGenerator** | In-memory | Local batch iteration |
| **KafkaStreamingGenerator** | Apache Kafka | Distributed event streaming |
| **WebSocketStreamingGenerator** | WebSocket | Real-time web applications |

## StreamingGenerator

Memory-efficient batch generation with iteration support.

```python
from genesis.streaming import StreamingGenerator, StreamingConfig

# Configure streaming
config = StreamingConfig(
    batch_size=1000,
    max_batches=100,
    delay_between_batches=0.1,  # seconds
)

# Create and fit generator
generator = StreamingGenerator(config=config)
generator.fit(training_data, method="gaussian_copula")

# Iterate over batches
for batch in generator.generate_stream():
    process_batch(batch)
    print(f"Processed {len(batch)} records")
```

### Batch Iterator

For more control over iteration:

```python
from genesis.streaming import BatchIterator

iterator = BatchIterator(
    generator=my_generator,
    total_samples=100000,
    batch_size=5000,
)

print(f"Total batches: {len(iterator)}")

for i, batch in enumerate(iterator):
    save_to_database(batch)
    print(f"Batch {i+1}/{len(iterator)} complete")
```

### Streaming Statistics

Track generation performance:

```python
generator = StreamingGenerator()
generator.fit(data)

# Generate with stats tracking
for batch in generator.generate_stream():
    process(batch)

# Check statistics
stats = generator.stats
print(f"Total samples: {stats.total_samples}")
print(f"Total batches: {stats.total_batches}")
print(f"Samples/second: {stats.samples_per_second:.2f}")
print(f"Elapsed time: {stats.elapsed_time:.2f}s")
```

## KafkaStreamingGenerator

Stream synthetic data to Apache Kafka topics for distributed processing.

```python
from genesis.streaming import KafkaStreamingGenerator

# Configure Kafka generator
generator = KafkaStreamingGenerator(
    bootstrap_servers="localhost:9092",
    topic="synthetic-data",
    batch_size=500,
    key_column="customer_id",  # Optional: use column as message key
)

# Fit to training data
generator.fit(training_data)

# Start streaming (runs continuously)
generator.start_streaming(
    samples_per_second=1000,
    max_samples=1000000,  # Optional limit
)
```

### Kafka Configuration

```python
generator = KafkaStreamingGenerator(
    bootstrap_servers="broker1:9092,broker2:9092",
    topic="synthetic-events",
    batch_size=100,
    
    # Producer configuration
    producer_config={
        "acks": "all",
        "retries": 3,
        "linger_ms": 10,
        "compression_type": "gzip",
    },
    
    # Serialization
    value_serializer="json",  # or "avro", "protobuf"
    schema_registry_url="http://schema-registry:8081",  # for Avro
)
```

### Message Format

Messages are serialized as JSON by default:

```json
{
  "customer_id": "C12345",
  "age": 34,
  "income": 75000.50,
  "region": "West",
  "_generated_at": "2026-01-28T10:30:00Z",
  "_batch_id": "batch-001"
}
```

## WebSocketStreamingGenerator

Stream data to web applications via WebSocket connections.

```python
from genesis.streaming import WebSocketStreamingGenerator

# Create WebSocket generator
generator = WebSocketStreamingGenerator(
    host="0.0.0.0",
    port=8765,
    batch_size=10,
)

# Fit generator
generator.fit(training_data)

# Start WebSocket server
await generator.start_server()
```

### Client Connection (JavaScript)

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onmessage = (event) => {
    const batch = JSON.parse(event.data);
    batch.records.forEach(record => {
        displayRecord(record);
    });
};

// Request specific conditions
ws.send(JSON.stringify({
    action: 'generate',
    count: 100,
    conditions: {
        age: { gte: 21 },
        region: 'West'
    }
}));
```

### Python Client

```python
import asyncio
import websockets
import json

async def consume_synthetic_data():
    async with websockets.connect('ws://localhost:8765') as ws:
        # Request data
        await ws.send(json.dumps({
            "action": "generate",
            "count": 1000,
            "batch_size": 100,
        }))
        
        # Receive batches
        while True:
            message = await ws.recv()
            data = json.loads(message)
            
            if data.get("status") == "complete":
                break
                
            process_batch(data["records"])

asyncio.run(consume_synthetic_data())
```

## Async Generation

For async workflows without streaming infrastructure:

```python
from genesis.streaming import generate_to_queue
import asyncio

async def main():
    queue = asyncio.Queue()
    
    # Start generation in background
    task = asyncio.create_task(
        generate_to_queue(
            generator=my_generator,
            queue=queue,
            total_samples=10000,
            batch_size=500,
        )
    )
    
    # Consume from queue
    while True:
        batch = await queue.get()
        if batch is None:  # Sentinel for completion
            break
        await process_async(batch)
    
    await task

asyncio.run(main())
```

## Data Stream Processor

Process streaming data with transformations:

```python
from genesis.streaming import DataStreamProcessor

processor = DataStreamProcessor(
    generator=my_generator,
    transformers=[
        lambda df: df.assign(processed_at=pd.Timestamp.now()),
        lambda df: df[df["value"] > 0],  # Filter
    ],
)

for batch in processor.process(total_samples=10000):
    save_batch(batch)
```

## Configuration Reference

### StreamingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | int | 1000 | Records per batch |
| `max_batches` | int | None | Maximum batches (None=unlimited) |
| `delay_between_batches` | float | 0.0 | Seconds between batches |
| `include_metadata` | bool | True | Add generation metadata |

### KafkaStreamingGenerator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bootstrap_servers` | str | required | Kafka broker addresses |
| `topic` | str | required | Target topic name |
| `batch_size` | int | 100 | Records per message batch |
| `key_column` | str | None | Column for message key |
| `producer_config` | dict | {} | Kafka producer settings |

### WebSocketStreamingGenerator

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | "0.0.0.0" | Bind address |
| `port` | int | 8765 | WebSocket port |
| `batch_size` | int | 100 | Records per message |
| `max_connections` | int | 100 | Maximum concurrent clients |

## Best Practices

### 1. Choose Right Batch Size
```python
# Small batches for low-latency
config = StreamingConfig(batch_size=10)  # Real-time dashboards

# Large batches for throughput
config = StreamingConfig(batch_size=10000)  # Bulk processing
```

### 2. Handle Backpressure
```python
generator = KafkaStreamingGenerator(
    topic="synthetic-data",
    producer_config={
        "max_block_ms": 60000,  # Wait up to 60s if buffer full
        "buffer_memory": 67108864,  # 64MB buffer
    }
)
```

### 3. Monitor Generation Stats
```python
import logging

for batch in generator.generate_stream():
    process(batch)
    
    if generator.stats.total_batches % 100 == 0:
        logging.info(
            f"Progress: {generator.stats.total_samples} samples, "
            f"{generator.stats.samples_per_second:.0f} samples/sec"
        )
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator
from genesis.streaming import StreamingGenerator, StreamingConfig

# Prepare data
data = pd.read_csv("events.csv")

# Train generator
base_generator = SyntheticGenerator(method="ctgan")
base_generator.fit(data)

# Configure streaming
config = StreamingConfig(
    batch_size=5000,
    delay_between_batches=0.5,
)

streaming = StreamingGenerator(config=config)
streaming._generator = base_generator

# Stream to file
with open("synthetic_events.jsonl", "w") as f:
    for batch in streaming.generate_stream():
        for _, row in batch.iterrows():
            f.write(row.to_json() + "\n")
        
        print(f"Written {streaming.stats.total_samples} records")
        
        if streaming.stats.total_samples >= 1000000:
            break

print(f"Complete: {streaming.stats.total_samples} records in {streaming.stats.elapsed_time:.1f}s")
```
