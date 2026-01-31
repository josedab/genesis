# Genesis v1.3.0 Architecture

This document describes the architecture of the new features introduced in Genesis v1.3.0.

## Feature Overview

```mermaid
graph TB
    subgraph "Genesis v1.3.0 Features"
        CG[Conditional Generation]
        ST[Streaming]
        FD[Federated Learning]
        LN[Data Lineage]
        DB[Quality Dashboard]
        NL[Natural Language API]
    end

    subgraph "Core Genesis"
        GEN[SyntheticGenerator]
        EVAL[QualityEvaluator]
        PRIV[PrivacyEngine]
    end

    CG --> GEN
    ST --> GEN
    FD --> GEN
    LN --> GEN
    LN --> EVAL
    DB --> EVAL
    NL --> CG
    NL --> GEN
```

## Conditional Generation Architecture

```mermaid
classDiagram
    class Condition {
        +column: str
        +operator: str
        +value: Any
        +apply(df) DataFrame
        +to_dict() dict
    }

    class ConditionSet {
        +conditions: List[Condition]
        +add(condition)
        +apply(df) DataFrame
        +from_dict(d) ConditionSet
    }

    class ConditionBuilder {
        -_conditions: List[Condition]
        -_current_column: str
        +where(column) ConditionBuilder
        +eq(value) ConditionBuilder
        +gte(value) ConditionBuilder
        +between(min, max) ConditionBuilder
        +build() ConditionSet
    }

    class ConditionalSampler {
        -_base_data: DataFrame
        +sample(conditions, n_samples) DataFrame
        +estimate_feasibility(conditions) float
    }

    class GuidedConditionalSampler {
        -_strategy: str
        -_column_stats: dict
        +fit(data)
        +sample(generator_fn, n_samples, conditions) DataFrame
    }

    ConditionSet "1" *-- "*" Condition
    ConditionBuilder ..> ConditionSet : builds
    ConditionalSampler --> ConditionSet : uses
    GuidedConditionalSampler --> ConditionSet : uses
```

### Sampling Strategies

```mermaid
flowchart LR
    subgraph "Iterative Refinement"
        IR1[Generate Batch] --> IR2[Apply Conditions]
        IR2 --> IR3{Enough Samples?}
        IR3 -->|No| IR1
        IR3 -->|Yes| IR4[Return Results]
    end

    subgraph "Importance Sampling"
        IS1[Analyze Condition Stats] --> IS2[Compute Weights]
        IS2 --> IS3[Oversample Regions]
        IS3 --> IS4[Filter & Reweight]
        IS4 --> IS5[Return Results]
    end
```

## Streaming Architecture

```mermaid
classDiagram
    class StreamingConfig {
        +batch_size: int
        +max_batches: int
        +delay_between_batches: float
    }

    class StreamingStats {
        +total_samples: int
        +total_batches: int
        +start_time: float
        +samples_per_second() float
    }

    class StreamingGenerator {
        -_config: StreamingConfig
        -_generator: BaseGenerator
        +fit(data, method)
        +generate_stream() Iterator
        +stats: StreamingStats
    }

    class KafkaStreamingGenerator {
        -_bootstrap_servers: str
        -_topic: str
        -_producer: KafkaProducer
        +start_streaming(samples_per_second)
        +stop()
    }

    class WebSocketStreamingGenerator {
        -_host: str
        -_port: int
        +start_server()
        +broadcast(data)
    }

    StreamingGenerator --> StreamingConfig
    StreamingGenerator --> StreamingStats
    KafkaStreamingGenerator --|> StreamingGenerator
    WebSocketStreamingGenerator --|> StreamingGenerator
```

### Data Flow

```mermaid
flowchart LR
    subgraph "Streaming Pipeline"
        GEN[Generator] --> BATCH[Batch Iterator]
        BATCH --> TRANSFORM[Transform]
        TRANSFORM --> OUTPUT{Output}
        OUTPUT --> KAFKA[Kafka Topic]
        OUTPUT --> WS[WebSocket]
        OUTPUT --> FILE[File/Queue]
    end
```

## Federated Learning Architecture

```mermaid
flowchart TB
    subgraph "Federated Training"
        AGG[Aggregator]
        
        subgraph "Site A"
            SA_DATA[(Local Data)]
            SA_MODEL[Local Model]
            SA_DATA --> SA_MODEL
        end
        
        subgraph "Site B"
            SB_DATA[(Local Data)]
            SB_MODEL[Local Model]
            SB_DATA --> SB_MODEL
        end
        
        subgraph "Site C"
            SC_DATA[(Local Data)]
            SC_MODEL[Local Model]
            SC_DATA --> SC_MODEL
        end
        
        SA_MODEL -->|Parameters| AGG
        SB_MODEL -->|Parameters| AGG
        SC_MODEL -->|Parameters| AGG
        
        AGG -->|Global Model| SYNTH[Synthetic Generator]
    end
```

### Class Hierarchy

```mermaid
classDiagram
    class SiteConfig {
        +name: str
        +weight: float
        +privacy_budget: float
        +min_samples: int
    }

    class DataSite {
        +name: str
        +n_samples: int
        +initialize(method)
        +train_local() dict
    }

    class ModelAggregator {
        +aggregate(site_params) AggregatedModel
    }

    class SecureAggregator {
        -noise_scale: float
        -min_sites: int
        +aggregate(site_params) AggregatedModel
    }

    class FederatedGenerator {
        -sites: List[DataSite]
        -aggregator: ModelAggregator
        +add_site(site)
        +train(rounds) AggregatedModel
        +generate(n_samples) DataFrame
    }

    class FederatedTrainingSimulator {
        -n_sites: int
        +setup_from_data(data)
        +setup_non_iid(data, column)
        +simulate_training(n_rounds) dict
    }

    SecureAggregator --|> ModelAggregator
    FederatedGenerator --> ModelAggregator
    FederatedGenerator "1" *-- "*" DataSite
    DataSite --> SiteConfig
```

## Data Lineage Architecture

```mermaid
classDiagram
    class LineageBlock {
        +block_id: str
        +block_type: str
        +timestamp: datetime
        +previous_hash: str
        +data_hash: str
        +content: dict
        +compute_hash() str
        +verify() bool
    }

    class LineageChain {
        -blocks: List[LineageBlock]
        +add_source_block(data, name)
        +add_transform_block(operation, params)
        +add_generation_block(method, n_samples)
        +add_quality_check(metrics, passed)
        +verify() bool
        +export(path)
        +load(path) LineageChain
    }

    class DataLineage {
        +sources: dict
        +transformations: list
        +record_source(name, data)
        +record_transformation(name, input_name)
        +record_generation(name, method, n_samples)
        +create_manifest() LineageManifest
    }

    LineageChain "1" *-- "*" LineageBlock
```

### Blockchain-Style Hash Chain

```mermaid
flowchart LR
    subgraph "Lineage Chain"
        B0[Genesis Block<br/>hash: 000...] --> B1[Source Block<br/>prev: 000...]
        B1 --> B2[Transform Block<br/>prev: hash1]
        B2 --> B3[Generation Block<br/>prev: hash2]
        B3 --> B4[Quality Block<br/>prev: hash3]
    end

    style B0 fill:#e1f5fe
    style B1 fill:#fff3e0
    style B2 fill:#f3e5f5
    style B3 fill:#e8f5e9
    style B4 fill:#fce4ec
```

## Quality Dashboard Architecture

```mermaid
classDiagram
    class QualityDashboard {
        -real_data: DataFrame
        -synthetic_data: DataFrame
        +compute_metrics() dict
        +generate_html_report() str
        +generate_plotly_figures() dict
        +save_report(path)
        +save_pdf(path)
    }

    class InteractiveDashboard {
        -app: FastAPI
        -host: str
        -port: int
        +run()
        +update_synthetic(data)
    }

    class QualityEvaluator {
        +evaluate() QualityReport
    }

    QualityDashboard --> QualityEvaluator
    InteractiveDashboard --> QualityDashboard
```

### Dashboard Components

```mermaid
flowchart TB
    subgraph "Quality Dashboard"
        INPUT[Real + Synthetic Data]
        
        subgraph "Metrics Engine"
            STAT[Statistical Tests]
            ML[ML Utility]
            PRIV[Privacy Analysis]
        end
        
        subgraph "Visualization"
            DIST[Distribution Plots]
            CORR[Correlation Heatmaps]
            PCA[PCA Projection]
            METRICS[Metrics Summary]
        end
        
        subgraph "Output"
            HTML[HTML Report]
            PDF[PDF Export]
            API[REST API]
        end
        
        INPUT --> STAT
        INPUT --> ML
        INPUT --> PRIV
        
        STAT --> DIST
        STAT --> CORR
        ML --> METRICS
        PRIV --> METRICS
        INPUT --> PCA
        
        DIST --> HTML
        CORR --> HTML
        PCA --> HTML
        METRICS --> HTML
        
        HTML --> PDF
        DIST --> API
        METRICS --> API
    end
```

## Natural Language API Architecture

```mermaid
sequenceDiagram
    participant User
    participant CLI/API
    participant IntentParser
    participant ConditionBuilder
    participant Generator
    
    User->>CLI/API: "Generate 1000 customers over 50 in California"
    CLI/API->>IntentParser: Parse request
    IntentParser->>IntentParser: Extract: n=1000, conditions
    IntentParser->>ConditionBuilder: Build conditions
    ConditionBuilder-->>IntentParser: ConditionSet
    IntentParser->>Generator: generate(1000, conditions)
    Generator-->>CLI/API: DataFrame
    CLI/API-->>User: Synthetic data
```

## Module Dependencies

```mermaid
flowchart TB
    subgraph "Public API"
        GENESIS[genesis]
        CLI[genesis.cli]
        API[genesis.api]
    end
    
    subgraph "Generation"
        COND[conditional]
        STREAM[streaming]
        FED[federated]
        TAB[tabular]
        TS[timeseries]
        TEXT[text]
    end
    
    subgraph "Quality"
        EVAL[evaluation]
        DASH[dashboard]
        PRIV[privacy]
    end
    
    subgraph "Infrastructure"
        LINE[lineage]
        CORE[core]
    end
    
    GENESIS --> COND
    GENESIS --> STREAM
    GENESIS --> FED
    CLI --> COND
    CLI --> DASH
    API --> COND
    
    COND --> TAB
    STREAM --> TAB
    FED --> TAB
    
    DASH --> EVAL
    LINE --> CORE
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph "Production Deployment"
        LB[Load Balancer]
        
        subgraph "API Tier"
            API1[Genesis API]
            API2[Genesis API]
            API3[Genesis API]
        end
        
        subgraph "Worker Tier"
            W1[Generation Worker]
            W2[Generation Worker]
            GPU[GPU Worker]
        end
        
        subgraph "Storage"
            REDIS[(Redis Cache)]
            S3[(S3/MinIO)]
            PG[(PostgreSQL)]
        end
        
        subgraph "Streaming"
            KAFKA[Kafka]
            WS[WebSocket Gateway]
        end
        
        LB --> API1
        LB --> API2
        LB --> API3
        
        API1 --> REDIS
        API2 --> REDIS
        API3 --> REDIS
        
        API1 --> W1
        API2 --> W2
        API3 --> GPU
        
        W1 --> S3
        W2 --> S3
        GPU --> S3
        
        W1 --> KAFKA
        W2 --> KAFKA
        
        API1 --> WS
        API2 --> WS
    end
```

## Security Considerations

### Federated Learning Security

```mermaid
flowchart LR
    subgraph "Security Layers"
        DP[Differential Privacy]
        SA[Secure Aggregation]
        TLS[TLS Encryption]
        AUTH[Authentication]
    end
    
    subgraph "Data Flow"
        LOCAL[Local Data] -->|Never Shared| SITE[Site]
        SITE -->|Encrypted Params| AGG[Aggregator]
        AGG -->|DP Noise Added| GLOBAL[Global Model]
    end
    
    DP --> AGG
    SA --> AGG
    TLS --> SITE
    AUTH --> SITE
```

### Lineage Integrity

```mermaid
flowchart TB
    subgraph "Tamper Detection"
        BLOCK[New Block]
        HASH[SHA-256 Hash]
        PREV[Previous Hash]
        VERIFY[Verification]
        
        BLOCK --> HASH
        PREV --> HASH
        HASH --> VERIFY
        VERIFY -->|Valid| ACCEPT[Accept Block]
        VERIFY -->|Invalid| REJECT[Reject Chain]
    end
```
