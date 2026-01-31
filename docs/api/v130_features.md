# API Reference: v1.3.0 Features

This document provides API reference for features introduced in Genesis v1.3.0.

## Conditional Generation

### `genesis.generators.conditional`

#### `class Condition`

Represents a single filter condition.

```python
Condition(column: str, operator: str, value: Any)
```

**Parameters:**
- `column` (str): Column name to filter
- `operator` (str): Comparison operator (`eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `not_in`, `between`, `like`)
- `value` (Any): Value to compare against

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `apply(df)` | `DataFrame` | Apply condition to DataFrame |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(d)` | `Condition` | Deserialize from dictionary |

---

#### `class ConditionSet`

Collection of conditions applied together (AND logic).

```python
ConditionSet(conditions: List[Condition] = None)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `add(condition)` | `None` | Add a condition |
| `apply(df)` | `DataFrame` | Apply all conditions |
| `to_dict()` | `dict` | Serialize to dictionary |
| `from_dict(d)` | `ConditionSet` | Create from dict (see below) |

**`from_dict` format:**
```python
# Simple equality
{"age": 30}

# With operators
{"age": (">=", 21), "income": (">", 50000)}

# Between
{"age": ("between", (18, 65))}

# In list
{"country": ("in", ["US", "UK", "CA"])}
```

---

#### `class ConditionBuilder`

Fluent API for building conditions.

```python
ConditionBuilder()
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `where(column)` | `ConditionBuilder` | Start condition for column |
| `eq(value)` | `ConditionBuilder` | Equals |
| `ne(value)` | `ConditionBuilder` | Not equals |
| `gt(value)` | `ConditionBuilder` | Greater than |
| `gte(value)` | `ConditionBuilder` | Greater than or equal |
| `lt(value)` | `ConditionBuilder` | Less than |
| `lte(value)` | `ConditionBuilder` | Less than or equal |
| `in_(values)` | `ConditionBuilder` | In list |
| `not_in(values)` | `ConditionBuilder` | Not in list |
| `between(min, max)` | `ConditionBuilder` | Range inclusive |
| `build()` | `ConditionSet` | Build final condition set |

**Example:**
```python
conditions = (
    ConditionBuilder()
    .where("age").gte(21).lte(65)
    .where("status").eq("active")
    .build()
)
```

---

#### `class ConditionalSampler`

Basic rejection sampling with conditions.

```python
ConditionalSampler(base_data: pd.DataFrame)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `sample` | `(conditions, n_samples, seed=None)` | `DataFrame` | Sample data matching conditions |
| `estimate_feasibility` | `(conditions)` | `float` | Estimate match probability |

---

#### `class GuidedConditionalSampler`

Intelligent conditional sampling with strategies.

```python
GuidedConditionalSampler(
    strategy: str = "iterative_refinement",
    max_iterations: int = 100,
    batch_multiplier: float = 5.0
)
```

**Parameters:**
- `strategy` (str): `"iterative_refinement"` or `"importance_sampling"`
- `max_iterations` (int): Maximum sampling attempts
- `batch_multiplier` (float): Oversample factor

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(data)` | `self` | Fit to training data |
| `sample` | `(generator_fn, n_samples, conditions, seed=None)` | `DataFrame` | Generate conditioned samples |

---

## Streaming

### `genesis.streaming`

#### `class StreamingConfig`

Configuration for streaming generation.

```python
StreamingConfig(
    batch_size: int = 1000,
    max_batches: Optional[int] = None,
    delay_between_batches: float = 0.0,
    include_metadata: bool = True
)
```

---

#### `class StreamingStats`

Statistics for streaming operations.

```python
StreamingStats()
```

**Attributes:**
- `total_samples` (int): Total generated samples
- `total_batches` (int): Total batches generated
- `start_time` (float): Generation start time
- `elapsed_time` (float): Time since start

**Properties:**
- `samples_per_second` (float): Generation throughput

---

#### `class StreamingGenerator`

Memory-efficient batch generation.

```python
StreamingGenerator(config: StreamingConfig = None)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(data, method="gaussian_copula", **kwargs)` | `self` | Fit generator |
| `generate` | `(n_samples)` | `DataFrame` | Generate samples |
| `generate_stream` | `()` | `Iterator[DataFrame]` | Iterate over batches |

**Properties:**
- `stats` (StreamingStats): Generation statistics

---

#### `class KafkaStreamingGenerator`

Stream to Apache Kafka.

```python
KafkaStreamingGenerator(
    bootstrap_servers: str,
    topic: str,
    batch_size: int = 100,
    key_column: Optional[str] = None,
    producer_config: dict = None,
    value_serializer: str = "json"
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(data, **kwargs)` | `self` | Fit generator |
| `start_streaming` | `(samples_per_second, max_samples=None)` | `None` | Start streaming |
| `stop` | `()` | `None` | Stop streaming |

---

#### `class WebSocketStreamingGenerator`

Stream via WebSocket.

```python
WebSocketStreamingGenerator(
    host: str = "0.0.0.0",
    port: int = 8765,
    batch_size: int = 100,
    max_connections: int = 100
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `fit` | `(data, **kwargs)` | `self` | Fit generator |
| `start_server` | `()` | `Coroutine` | Start WebSocket server |
| `broadcast` | `(data)` | `None` | Send to all clients |

---

#### `class BatchIterator`

Iterate over generated batches.

```python
BatchIterator(
    generator: BaseGenerator,
    total_samples: int,
    batch_size: int = 1000
)
```

**Methods:**
- `__iter__()`: Returns iterator
- `__len__()`: Returns number of batches

---

#### `async def generate_to_queue`

Generate samples to an async queue.

```python
async def generate_to_queue(
    generator: BaseGenerator,
    queue: asyncio.Queue,
    total_samples: int,
    batch_size: int = 1000
) -> None
```

---

## Federated Learning

### `genesis.federated`

#### `class SiteConfig`

Configuration for a data site.

```python
SiteConfig(
    name: str,
    weight: float = 1.0,
    privacy_budget: float = 1.0,
    min_samples: int = 100
)
```

---

#### `class DataSite`

Represents a single federated data site.

```python
DataSite(
    name: str,
    data: pd.DataFrame,
    config: SiteConfig = None
)
```

**Properties:**
- `name` (str): Site identifier
- `n_samples` (int): Number of local samples

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `initialize` | `(method="gaussian_copula")` | `None` | Initialize local model |
| `train_local` | `()` | `dict` | Train and return parameters |

---

#### `class AggregatedModel`

Result of federated aggregation.

```python
AggregatedModel(
    parameters: dict,
    n_sites: int,
    total_samples: int,
    round_number: int,
    metadata: dict = None
)
```

---

#### `class ModelAggregator`

Basic parameter aggregation.

```python
ModelAggregator()
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `aggregate` | `(site_params: List[dict])` | `AggregatedModel` | Aggregate parameters |

---

#### `class SecureAggregator`

Aggregation with differential privacy.

```python
SecureAggregator(
    noise_scale: float = 0.1,
    min_sites: int = 2,
    clip_threshold: float = 1.0,
    epsilon: Optional[float] = None,
    delta: float = 1e-5
)
```

Inherits from `ModelAggregator`.

---

#### `class FederatedGenerator`

Coordinate federated training.

```python
FederatedGenerator(
    aggregator: ModelAggregator = None,
    method: str = "gaussian_copula"
)
```

**Properties:**
- `sites` (List[DataSite]): Registered sites
- `is_trained` (bool): Training status
- `method` (str): Generation method

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_site` | `(site: DataSite)` | `None` | Register a site |
| `train` | `(rounds: int = 1)` | `AggregatedModel` | Run federated training |
| `generate` | `(n_samples, strategy="proportional")` | `DataFrame` | Generate synthetic data |

---

#### `class FederatedTrainingSimulator`

Simulate federated scenarios.

```python
FederatedTrainingSimulator(n_sites: int = 3)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `setup_from_data` | `(data)` | `None` | Partition data IID |
| `setup_non_iid` | `(data, partition_column, alpha=0.5)` | `None` | Partition non-IID |
| `simulate_training` | `(n_rounds, **kwargs)` | `dict` | Simulate training |
| `generate_synthetic` | `(n_samples)` | `DataFrame` | Generate from trained model |

---

#### `def create_federated_generator`

Convenience function for creating federated generator.

```python
def create_federated_generator(
    site_datasets: Dict[str, pd.DataFrame],
    method: str = "gaussian_copula",
    aggregator: ModelAggregator = None
) -> FederatedGenerator
```

---

## Data Lineage

### `genesis.lineage`

#### `class LineageBlock`

Single block in lineage chain.

```python
LineageBlock(
    block_type: str,
    content: dict,
    previous_hash: str = None,
    metadata: dict = None
)
```

**Properties:**
- `block_id` (str): Unique identifier
- `block_type` (str): Type of block
- `timestamp` (datetime): Creation time
- `previous_hash` (str): Hash of previous block
- `data_hash` (str): Hash of this block

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `compute_hash` | `()` | `str` | Compute SHA-256 hash |
| `verify` | `()` | `bool` | Verify block integrity |
| `to_dict` | `()` | `dict` | Serialize to dictionary |
| `from_dict` | `(d)` | `LineageBlock` | Deserialize |

---

#### `class LineageChain`

Blockchain-style immutable audit trail.

```python
LineageChain()
```

**Properties:**
- `blocks` (List[LineageBlock]): All blocks
- `latest_hash` (str): Hash of latest block

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_source_block` | `(data, name, metadata=None)` | `None` | Record data source |
| `add_transform_block` | `(operation, input_hash, parameters)` | `None` | Record transformation |
| `add_generation_block` | `(method, parameters, n_samples, quality_metrics=None)` | `None` | Record generation |
| `add_quality_check` | `(metrics, passed, threshold=None)` | `None` | Record quality check |
| `verify` | `()` | `bool` | Verify entire chain |
| `get_audit_trail` | `()` | `List[dict]` | Human-readable trail |
| `export` | `(path)` | `None` | Save to JSON |
| `load` | `(path)` | `LineageChain` | Load from JSON |

---

#### `class DataLineage`

Simple lineage tracking.

```python
DataLineage()
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `record_source` | `(name, data, description=None, tags=None)` | `None` | Record source |
| `record_transformation` | `(name, input_name, description=None)` | `None` | Record transform |
| `record_generation` | `(name, method, n_samples, parameters=None)` | `None` | Record generation |
| `create_manifest` | `()` | `LineageManifest` | Export manifest |

---

## Quality Dashboard

### `genesis.dashboard`

#### `class QualityDashboard`

Generate quality reports.

```python
QualityDashboard(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    name: str = "Quality Report",
    include_privacy: bool = True,
    include_ml_utility: bool = True
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `compute_metrics` | `()` | `dict` | Compute all metrics |
| `generate_html_report` | `()` | `str` | Generate HTML |
| `generate_plotly_figures` | `()` | `dict` | Generate Plotly figures |
| `save_report` | `(path)` | `None` | Save HTML to file |
| `save_pdf` | `(path)` | `None` | Export as PDF |

---

#### `class InteractiveDashboard`

Live dashboard server.

```python
InteractiveDashboard(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    host: str = "0.0.0.0",
    port: int = 8050,
    debug: bool = False
)
```

**Methods:**

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `run` | `()` | `None` | Start server |
| `update_synthetic` | `(data)` | `None` | Update synthetic data |

---

#### `def create_dashboard`

Convenience function.

```python
def create_dashboard(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    output_path: Optional[str] = None
) -> str
```

---

## CLI Commands

### `genesis chat`

Interactive natural language generation.

```bash
genesis chat [OPTIONS]
```

**Options:**
- `--data PATH`: Training data file
- `--model TEXT`: Generation method

---

### `genesis dashboard`

Generate quality dashboard.

```bash
genesis dashboard REAL_DATA SYNTHETIC_DATA [OPTIONS]
```

**Options:**
- `--output PATH`: Output file path
- `--serve`: Launch interactive server
- `--port INT`: Server port (default: 8050)

---

### `genesis discover`

Discover schema from data.

```bash
genesis discover DATA_PATH [OPTIONS]
```

**Options:**
- `--output PATH`: Output schema file
- `--format TEXT`: Output format (yaml, json, python)

---

## REST API Endpoints

### Natural Language Generation

#### `POST /v1/generate/natural-language`

Generate data from natural language description.

**Request:**
```json
{
  "prompt": "Generate 1000 customers over 50 in California",
  "data_context": "customers"
}
```

**Response:**
```json
{
  "data": [...],
  "interpretation": {
    "n_samples": 1000,
    "conditions": {"age": [">=", 50], "state": "California"}
  }
}
```

---

#### `POST /v1/generate/natural-language/clarify`

Request clarification for ambiguous prompts.

**Request:**
```json
{
  "prompt": "Generate some customer data",
  "session_id": "abc123"
}
```

**Response:**
```json
{
  "needs_clarification": true,
  "questions": [
    {"field": "n_samples", "question": "How many samples?"},
    {"field": "conditions", "question": "Any specific conditions?"}
  ]
}
```
