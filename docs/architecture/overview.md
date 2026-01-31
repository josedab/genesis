# Genesis Architecture Overview

This document provides a high-level architecture overview of the Genesis synthetic data generation platform.

## System Architecture

```mermaid
graph TB
    subgraph "User Interface Layer"
        CLI[CLI Interface]
        API[REST API]
        SDK[Python SDK]
        DASH[Dashboard]
    end
    
    subgraph "Core Engine"
        subgraph "Generators"
            CTGAN[CTGAN]
            TVAE[TVAE]
            GC[Gaussian Copula]
            TGAN[TimeGAN]
            LLM[LLM Text]
            IMG[Image Synth]
        end
        
        subgraph "Processing"
            TRANS[Transformers]
            CONST[Constraints]
            COND[Conditions]
        end
        
        subgraph "Evaluation"
            STAT[Statistical]
            PRIV[Privacy]
            ML[ML Utility]
        end
    end
    
    subgraph "Extension Layer"
        PLUGINS[Plugin System]
        TUNING[Auto-Tuner]
    end
    
    subgraph "Quality & Compliance"
        COMP[Privacy Certificates]
        DRIFT[Drift Detection]
        DEBUG[Debugger]
    end
    
    subgraph "Advanced Generation"
        ANOM[Anomaly Synthesis]
        CROSS[Cross-Modal]
        DIST[Distributed Training]
        FED[Federated Learning]
    end
    
    subgraph "Platform Services"
        SCHEMA[Schema Editor]
        MARKET[Marketplace]
        LINEAGE[Data Lineage]
        STREAM[Streaming]
    end
    
    subgraph "Infrastructure"
        STORAGE[(Storage)]
        QUEUE[Job Queue]
        CACHE[Cache]
    end
    
    CLI --> SDK
    API --> SDK
    DASH --> SDK
    
    SDK --> PLUGINS
    SDK --> TUNING
    PLUGINS --> CTGAN
    PLUGINS --> TVAE
    PLUGINS --> GC
    
    CTGAN --> TRANS
    TVAE --> TRANS
    GC --> TRANS
    
    TRANS --> CONST
    CONST --> COND
    
    COND --> STAT
    COND --> PRIV
    COND --> ML
    
    PRIV --> COMP
    STAT --> DRIFT
    STAT --> DEBUG
    
    DIST --> CTGAN
    DIST --> TVAE
    CROSS --> CTGAN
    CROSS --> LLM
    
    SCHEMA --> MARKET
    LINEAGE --> MARKET
    
    DIST --> STORAGE
    MARKET --> STORAGE
    STREAM --> QUEUE
```

## Data Flow

```mermaid
sequenceDiagram
    participant User
    participant SDK
    participant Generator
    participant Transformer
    participant Evaluator
    participant Output

    User->>SDK: fit(data, config)
    SDK->>Transformer: preprocess(data)
    Transformer->>Generator: fit(transformed_data)
    Generator-->>SDK: fitted model
    
    User->>SDK: generate(n_samples)
    SDK->>Generator: sample(n_samples)
    Generator->>Transformer: inverse_transform(samples)
    Transformer->>Evaluator: evaluate(real, synthetic)
    Evaluator-->>SDK: quality_report
    SDK-->>User: synthetic_data, report
```

## Module Dependency Graph

```mermaid
graph LR
    subgraph "Core (genesis.core)"
        BASE[base]
        CONFIG[config]
        TYPES[types]
        EXCEPT[exceptions]
    end
    
    subgraph "Generators (genesis.generators)"
        TAB[tabular]
        TS[timeseries]
        TXT[text]
        COND[conditional]
    end
    
    subgraph "Evaluation (genesis.evaluation)"
        EVAL[evaluator]
        METRICS[metrics]
        PRIVACY[privacy]
        REPORT[report]
    end
    
    subgraph "Extensions (genesis.*)"
        PLUG[plugins]
        TUNE[tuning]
        AGENT[agents]
    end
    
    subgraph "Platform (genesis.*)"
        FEED[federated]
        STRM[streaming]
        LINE[lineage]
        DISC[discovery]
    end
    
    CONFIG --> BASE
    TYPES --> BASE
    EXCEPT --> BASE
    
    BASE --> TAB
    BASE --> TS
    BASE --> TXT
    TAB --> COND
    
    TAB --> EVAL
    TS --> EVAL
    TXT --> EVAL
    EVAL --> METRICS
    EVAL --> PRIVACY
    METRICS --> REPORT
    
    BASE --> PLUG
    TAB --> TUNE
    EVAL --> TUNE
    
    TAB --> FEED
    TAB --> STRM
    EVAL --> LINE
```

## Component Details

### Core Engine

The core engine provides the fundamental building blocks:

| Component | Purpose | Key Classes |
|-----------|---------|-------------|
| `base` | Abstract generator interface | `BaseGenerator`, `SyntheticGenerator` |
| `config` | Configuration management | `GeneratorConfig`, `PrivacyConfig` |
| `types` | Type definitions | `ColumnType`, `GeneratorMethod` |
| `constraints` | Data constraints | `Constraint`, `ConstraintSet` |

### Generators

Each generator implements a different synthesis algorithm:

| Generator | Algorithm | Best For |
|-----------|-----------|----------|
| `CTGANGenerator` | Conditional GAN | Mixed-type tabular |
| `TVAEGenerator` | Variational Autoencoder | Continuous-heavy data |
| `GaussianCopulaGenerator` | Copula modeling | Statistical fidelity |
| `TimeGANGenerator` | Temporal GAN | Time series |
| `LLMTextGenerator` | Large Language Model | Text generation |

### Evaluation Pipeline

```mermaid
graph LR
    REAL[Real Data] --> STAT[Statistical Tests]
    SYNTH[Synthetic Data] --> STAT
    STAT --> KS[KS Test]
    STAT --> CORR[Correlation]
    STAT --> DIST[Distribution]
    
    REAL --> PRIV[Privacy Metrics]
    SYNTH --> PRIV
    PRIV --> REID[Re-identification]
    PRIV --> MIA[Membership Inference]
    PRIV --> DCR[Distance to Closest]
    
    REAL --> ML[ML Utility]
    SYNTH --> ML
    ML --> TSTR[Train Synth Test Real]
    ML --> TRTS[Train Real Test Synth]
    
    KS --> REPORT[Quality Report]
    CORR --> REPORT
    DIST --> REPORT
    REID --> REPORT
    MIA --> REPORT
    DCR --> REPORT
    TSTR --> REPORT
    TRTS --> REPORT
```

### Plugin Architecture

```mermaid
classDiagram
    class PluginRegistry {
        -_instance: PluginRegistry
        -_plugins: Dict
        +register(name, cls, type)
        +get(name, type)
        +list_plugins(type)
        +unregister(name, type)
    }
    
    class PluginInfo {
        +name: str
        +plugin_type: PluginType
        +cls: Type
        +description: str
        +version: str
    }
    
    class PluginType {
        <<enumeration>>
        GENERATOR
        TRANSFORMER
        EVALUATOR
        CONSTRAINT
        CALLBACK
    }
    
    class BaseGenerator {
        <<abstract>>
        +fit(data)
        +generate(n_samples)
        +quality_report()
    }
    
    PluginRegistry --> PluginInfo
    PluginInfo --> PluginType
    PluginRegistry --> BaseGenerator
```

### Distributed Training Architecture

```mermaid
graph TB
    subgraph "Coordinator"
        TRAINER[DistributedTrainer]
        SHARDER[DataSharder]
        AGG[Aggregator]
    end
    
    subgraph "Workers"
        W1[Worker 1<br/>GPU 0]
        W2[Worker 2<br/>GPU 1]
        W3[Worker 3<br/>GPU 2]
        W4[Worker 4<br/>GPU 3]
    end
    
    subgraph "Backend"
        RAY[Ray Cluster]
        DASK[Dask Cluster]
        MP[Multiprocessing]
    end
    
    TRAINER --> SHARDER
    SHARDER --> |shard 1| W1
    SHARDER --> |shard 2| W2
    SHARDER --> |shard 3| W3
    SHARDER --> |shard 4| W4
    
    W1 --> AGG
    W2 --> AGG
    W3 --> AGG
    W4 --> AGG
    
    RAY -.-> W1
    RAY -.-> W2
    DASK -.-> W3
    DASK -.-> W4
```

## Security Model

```mermaid
graph TB
    subgraph "Data Protection"
        DP[Differential Privacy]
        KANON[K-Anonymity]
        LDIV[L-Diversity]
    end
    
    subgraph "Compliance"
        GDPR[GDPR]
        HIPAA[HIPAA]
        CCPA[CCPA]
    end
    
    subgraph "Monitoring"
        DRIFT[Drift Detection]
        AUDIT[Audit Logging]
        CERT[Privacy Certificates]
    end
    
    DP --> GDPR
    DP --> HIPAA
    KANON --> GDPR
    LDIV --> HIPAA
    
    GDPR --> CERT
    HIPAA --> CERT
    CCPA --> CERT
    
    CERT --> AUDIT
    DRIFT --> AUDIT
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        CLI[CLI]
        JUPYTER[Jupyter]
        APP[Application]
    end
    
    subgraph "API Gateway"
        LB[Load Balancer]
        AUTH[Authentication]
        RATE[Rate Limiter]
    end
    
    subgraph "Service Layer"
        API1[API Pod 1]
        API2[API Pod 2]
        API3[API Pod 3]
    end
    
    subgraph "Worker Layer"
        WORKER1[Worker 1]
        WORKER2[Worker 2]
        WORKER3[Worker 3]
    end
    
    subgraph "Data Layer"
        REDIS[(Redis Cache)]
        POSTGRES[(PostgreSQL)]
        S3[(Object Storage)]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus]
        GRAF[Grafana]
        ALERT[Alertmanager]
    end
    
    CLI --> LB
    JUPYTER --> LB
    APP --> LB
    
    LB --> AUTH
    AUTH --> RATE
    RATE --> API1
    RATE --> API2
    RATE --> API3
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> WORKER1
    API2 --> WORKER2
    API3 --> WORKER3
    
    WORKER1 --> POSTGRES
    WORKER2 --> POSTGRES
    WORKER3 --> S3
    
    API1 --> PROM
    WORKER1 --> PROM
    PROM --> GRAF
    PROM --> ALERT
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.8+ |
| ML Framework | PyTorch, TensorFlow (optional) |
| Data Processing | Pandas, NumPy, SciPy |
| ML Libraries | scikit-learn, SDV |
| API Framework | FastAPI |
| Distributed | Ray, Dask |
| Caching | Redis |
| Database | PostgreSQL |
| Monitoring | Prometheus, Grafana |
| Container | Docker, Kubernetes |

## See Also

- [API Reference](../api/reference.md)
- [v1.2.0 Features](../api/v120_features.md)
- [v1.3.0 Features](../api/v130_features.md)
- [v1.4.0 Features](../api/v140_features.md)
- [Architecture Decision Records](../adr/README.md)
