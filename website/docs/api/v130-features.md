---
sidebar_position: 4
title: v1.3.0 Features
---

# v1.3.0 Features API Reference

API reference for features introduced in Genesis v1.3.0.

## Conditional Generation

Generate data matching specific conditions.

### GuidedConditionalSampler

```python
from genesis.generators.conditional import GuidedConditionalSampler

sampler = GuidedConditionalSampler(
    generator,                    # Fitted generator
    strategy='iterative_refinement'  # or 'importance_sampling'
)

# Generate with conditions
synthetic = sampler.sample(
    n_samples=1000,
    conditions={
        'status': 'active',
        'age': ('>', 25),
        'income': ('between', 50000, 100000)
    }
)
```

#### Condition Types

```python
# Exact match
{'column': 'value'}

# Greater than
{'column': ('>', value)}

# Less than
{'column': ('<', value)}

# Greater than or equal
{'column': ('>=', value)}

# Between
{'column': ('between', min_value, max_value)}

# In list
{'column': ('in', [value1, value2, value3])}
```

---

## Streaming Generation

Real-time synthetic data streaming.

### StreamingGenerator

```python
from genesis.generators.streaming import StreamingGenerator

streamer = StreamingGenerator(generator)

# Start streaming
for batch in streamer.stream(
    batch_size=100,
    interval_seconds=1.0
):
    process(batch)

# With conditions
for batch in streamer.stream(
    batch_size=100,
    conditions={'status': 'active'}
):
    process(batch)
```

### KafkaPublisher

```python
from genesis.generators.streaming import KafkaPublisher

publisher = KafkaPublisher(
    bootstrap_servers='localhost:9092',
    topic='synthetic-data'
)

streamer.stream_to(
    publisher,
    batch_size=100,
    interval_seconds=0.5
)
```

### WebSocketPublisher

```python
from genesis.generators.streaming import WebSocketPublisher

publisher = WebSocketPublisher(
    uri='ws://localhost:8765'
)

await streamer.stream_to_async(publisher, batch_size=100)
```

---

## Federated Learning

Train across distributed data without centralization.

### FederatedSynthesizer

```python
from genesis.generators.federated import FederatedSynthesizer

synthesizer = FederatedSynthesizer(
    method='ctgan',
    n_clients=5,
    rounds=10,
    aggregation='fedavg'  # 'fedavg' or 'weighted'
)

# Train across client datasets
synthesizer.fit(client_datasets)

# Generate
synthetic = synthesizer.generate(n_samples=1000)
```

### Client Configuration

```python
synthesizer = FederatedSynthesizer(
    method='ctgan',
    client_config={
        'local_epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.001
    },
    privacy={
        'differential_privacy': {'epsilon': 1.0}
    }
)
```

---

## Data Lineage Tracking

Track provenance of synthetic data.

### LineageChain

```python
from genesis.generators.lineage import LineageChain, LineageEvent

# Create chain
chain = LineageChain()

# Add events
chain.add_event(LineageEvent(
    event_type='data_loaded',
    source='customers.csv',
    metadata={'rows': 10000, 'columns': 15}
))

chain.add_event(LineageEvent(
    event_type='generator_fitted',
    source='ctgan',
    metadata={'epochs': 300}
))

chain.add_event(LineageEvent(
    event_type='data_generated',
    source='synthesis',
    metadata={'n_samples': 5000}
))

# Verify chain integrity
is_valid = chain.verify()

# Export
chain.to_json('lineage.json')
chain.to_html('lineage_report.html')
```

### LineageTracker

```python
from genesis.generators.lineage import LineageTracker

# Automatic tracking
tracker = LineageTracker()

with tracker.track() as ctx:
    generator = SyntheticGenerator(method='ctgan')
    generator.fit(data)
    synthetic = generator.generate(1000)
    
# Access lineage
print(tracker.chain.to_dict())
```

---

## Dashboard & Reporting

Generate quality reports.

### DashboardGenerator

```python
from genesis.generators.dashboard import DashboardGenerator

dashboard = DashboardGenerator(real_data, synthetic_data)

# Generate HTML report
dashboard.generate_html('report.html')

# Generate PDF report
dashboard.generate_pdf('report.pdf')

# Get metrics dictionary
metrics = dashboard.get_metrics()
```

### Report Sections

```python
dashboard = DashboardGenerator(
    real_data,
    synthetic_data,
    sections=[
        'summary',           # Overall quality scores
        'distributions',     # Per-column distributions
        'correlations',      # Correlation comparison
        'privacy',           # Privacy metrics
        'recommendations'    # Improvement suggestions
    ]
)
```

### Custom Visualizations

```python
# Add custom plots
dashboard.add_plot(
    'custom_analysis',
    plot_function,
    title='Custom Analysis'
)

dashboard.generate_html('report.html')
```

---

## Natural Language API

Query and generate data using natural language.

### NaturalLanguageInterface

```python
from genesis.api.natural_language import NaturalLanguageInterface

nli = NaturalLanguageInterface(generator)

# Natural language queries
synthetic = nli.query(
    "Generate 1000 customers from the US with income above 50000"
)

synthetic = nli.query(
    "Create 500 high-risk transactions with amounts between 10000 and 100000"
)
```

### Query Parsing

```python
# Parse query to structured conditions
parsed = nli.parse(
    "Generate 1000 active premium customers"
)

print(parsed)
# {
#     'n_samples': 1000,
#     'conditions': {
#         'status': 'active',
#         'tier': 'premium'
#     }
# }
```

---

## GPU Acceleration

Use GPU for faster training.

### GPU Configuration

```python
from genesis import SyntheticGenerator

# Automatic GPU detection
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda'}
)

# Specific GPU
generator = SyntheticGenerator(
    method='ctgan',
    config={'device': 'cuda:1'}
)
```

### BatchedGenerator

```python
from genesis.gpu import BatchedGenerator

generator = BatchedGenerator(
    method='ctgan',
    device='cuda',
    batch_size=10000,
    mixed_precision=True  # FP16 for faster training
)

generator.fit(large_data)
synthetic = generator.generate(1_000_000)
```

### Multi-GPU

```python
from genesis.gpu import DistributedGenerator

generator = DistributedGenerator(
    method='ctgan',
    devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
)

generator.fit(data)
synthetic = generator.generate(n_samples)
```

---

## v1.3.0 Changelog

### New Features
- **Conditional generation** with multiple sampling strategies
- **Streaming generation** with Kafka and WebSocket support
- **Federated learning** for distributed training
- **Data lineage tracking** with blockchain-style verification
- **Dashboard & reports** with HTML/PDF export
- **Natural language interface** for queries
- **GPU acceleration** with multi-GPU support

### Improvements
- 40% faster CTGAN training
- Reduced memory usage for large datasets
- Better handling of high-cardinality categories
- Improved privacy guarantees

### API Changes
- Added `conditions` parameter to `generate()`
- New `privacy` parameter in `SyntheticGenerator`
- Deprecated `categorical_columns` (use `discrete_columns`)

See [CHANGELOG.md](https://github.com/genesis/genesis/blob/main/CHANGELOG.md) for complete details.
