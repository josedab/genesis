# Genesis v1.4.0 Architecture

This document describes the architecture of the new features introduced in Genesis v1.4.0.

## Feature Overview

```mermaid
graph TB
    subgraph "Genesis v1.4.0 Features"
        AM[AutoML Synthesis]
        AUG[Synthetic Augmentation]
        PA[Privacy Attacks]
        LLM[LLM Schema Inference]
        DR[Drift Detection]
        VER[Dataset Versioning]
        GPU[GPU Acceleration]
        DOM[Domain Generators]
        PIP[Pipeline Builder]
    end

    subgraph "Core Genesis"
        GEN[SyntheticGenerator]
        EVAL[QualityEvaluator]
        PRIV[PrivacyEngine]
    end

    AM --> GEN
    AUG --> GEN
    PA --> PRIV
    LLM --> GEN
    DR --> EVAL
    DR --> GEN
    VER --> GEN
    GPU --> GEN
    DOM --> GEN
    PIP --> GEN
    PIP --> EVAL
```

## AutoML Synthesis Architecture

```mermaid
classDiagram
    class MetaFeatures {
        +n_rows: int
        +n_columns: int
        +numeric_ratio: float
        +categorical_ratio: float
        +missing_ratio: float
        +has_high_cardinality: bool
        +has_temporal: bool
        +max_correlation: float
    }

    class MetaFeatureExtractor {
        +extract(df) MetaFeatures
        -_analyze_types(df)
        -_analyze_correlations(df)
        -_analyze_cardinality(df)
    }

    class MethodRecommendation {
        +method: GenerationMethod
        +confidence: float
        +reason: str
    }

    class SelectionResult {
        +recommended_method: GenerationMethod
        +confidence: float
        +reason: str
        +all_recommendations: List
    }

    class MethodSelector {
        +prefer_speed: bool
        +prefer_quality: bool
        +select(features) SelectionResult
        -_score_method(method, features)
    }

    class AutoMLSynthesizer {
        +selected_method: GenerationMethod
        +selection_confidence: float
        +fit(df) AutoMLSynthesizer
        +generate(n_samples) DataFrame
    }

    MetaFeatureExtractor --> MetaFeatures
    MethodSelector --> SelectionResult
    SelectionResult --> MethodRecommendation
    AutoMLSynthesizer --> MetaFeatureExtractor
    AutoMLSynthesizer --> MethodSelector
```

## Data Augmentation Architecture

```mermaid
sequenceDiagram
    participant User
    participant Planner as AugmentationPlanner
    participant Augmenter as SyntheticAugmenter
    participant Generator as SyntheticGenerator

    User->>Planner: analyze(df, target)
    Planner-->>User: AugmentationPlan
    
    User->>Augmenter: fit(df, target)
    Augmenter->>Generator: fit(minority_data)
    
    User->>Augmenter: augment(target_ratio)
    loop For each minority class
        Augmenter->>Generator: generate(n_needed)
        Generator-->>Augmenter: synthetic_samples
    end
    Augmenter-->>User: balanced_df
```

## Privacy Attack Testing Architecture

```mermaid
classDiagram
    class AttackResult {
        +accuracy: float
        +risk_level: str
        +details: dict
    }

    class MembershipInferenceAttack {
        +run(real, synthetic, holdout) MembershipResult
        -_train_attack_model()
        -_evaluate_attack()
    }

    class AttributeInferenceAttack {
        +run(real, synthetic, sensitive, known) AttributeResult
        -_build_inference_model()
        -_measure_leakage()
    }

    class ReidentificationAttack {
        +run(real, synthetic, quasi_ids) ReidentificationResult
        -_find_matches()
        -_calculate_risk()
    }

    class PrivacyAttackTester {
        +sensitive_columns: List
        +quasi_identifiers: List
        +run_all_attacks() PrivacyAuditReport
    }

    class PrivacyAuditReport {
        +overall_risk: str
        +passed: bool
        +membership_result: MembershipResult
        +attribute_results: dict
        +reidentification_result: ReidentificationResult
        +to_html(path)
        +to_json(path)
    }

    PrivacyAttackTester --> MembershipInferenceAttack
    PrivacyAttackTester --> AttributeInferenceAttack
    PrivacyAttackTester --> ReidentificationAttack
    PrivacyAttackTester --> PrivacyAuditReport
    MembershipInferenceAttack --> AttackResult
    AttributeInferenceAttack --> AttackResult
    ReidentificationAttack --> AttackResult
```

## Drift Detection Architecture

```mermaid
flowchart LR
    subgraph Detection
        B[Baseline Data] --> D{DriftDetector}
        C[Current Data] --> D
        D --> R[DriftReport]
    end

    subgraph Metrics
        D --> KS[KS Test]
        D --> JS[JS Divergence]
        D --> PSI[PSI Score]
    end

    subgraph Response
        R --> |drift detected| G[DriftAwareGenerator]
        G --> |adapt| S[Synthetic Data]
    end
```

## Dataset Versioning Architecture

```mermaid
classDiagram
    class DatasetRepository {
        +path: str
        +init(path) DatasetRepository
        +commit(df, message) Commit
        +log() List~Commit~
        +branch(name)
        +checkout(ref)
        +merge(branch)
        +tag(name)
        +diff(ref1, ref2) DatasetDiff
    }

    class Commit {
        +hash: str
        +message: str
        +timestamp: datetime
        +metadata: dict
        +parent: str
        +load_data() DataFrame
    }

    class Branch {
        +name: str
        +head: str
        +is_current: bool
    }

    class Tag {
        +name: str
        +commit_hash: str
        +message: str
    }

    class DatasetDiff {
        +rows_added: int
        +rows_removed: int
        +columns_changed: List
        +column_changes: dict
    }

    DatasetRepository --> Commit
    DatasetRepository --> Branch
    DatasetRepository --> Tag
    DatasetRepository --> DatasetDiff
```

## Pipeline Builder Architecture

```mermaid
flowchart TB
    subgraph Builder
        PB[PipelineBuilder] --> |source| SN[SourceNode]
        PB --> |transform| TN[TransformNode]
        PB --> |synthesize| GN[SynthesizeNode]
        PB --> |evaluate| EN[EvaluateNode]
        PB --> |sink| KN[SinkNode]
    end

    subgraph Execution
        PB --> |build| P[Pipeline]
        P --> |validate| V{Valid?}
        V --> |yes| EX[PipelineExecutor]
        EX --> |topological sort| TS[Execution Order]
        TS --> |execute| R[Results]
    end

    subgraph Nodes
        SN --> |data| TN
        TN --> |data| GN
        GN --> |data| EN
        EN --> |data| KN
    end
```

## Domain Generators Architecture

```mermaid
classDiagram
    class DomainGenerator {
        <<abstract>>
        +domain: str
        +fit(data)
        +generate(n_samples)
    }

    class HealthcareGenerator {
        +generate_patient_cohort()
        +generate_clinical_events()
        +generate_lab_results()
        +generate_medications()
    }

    class FinanceGenerator {
        +generate_transactions()
        +generate_accounts()
        +generate_credit_profiles()
        +generate_fraud_scenarios()
    }

    class RetailGenerator {
        +generate_customers()
        +generate_orders()
        +generate_products()
        +generate_ecommerce_dataset()
    }

    DomainGenerator <|-- HealthcareGenerator
    DomainGenerator <|-- FinanceGenerator
    DomainGenerator <|-- RetailGenerator
```

## GPU Acceleration Architecture

```mermaid
flowchart TB
    subgraph "Single GPU"
        D1[Data] --> BG[BatchedGenerator]
        BG --> |batch 1| G1[GPU Process]
        BG --> |batch 2| G1
        BG --> |batch N| G1
        G1 --> O1[Synthetic Data]
    end

    subgraph "Multi-GPU"
        D2[Data] --> MG[MultiGPUGenerator]
        MG --> |shard 1| GPU0[GPU 0]
        MG --> |shard 2| GPU1[GPU 1]
        MG --> |shard N| GPUN[GPU N]
        GPU0 --> AGG[Aggregator]
        GPU1 --> AGG
        GPUN --> AGG
        AGG --> O2[Synthetic Data]
    end
```

## LLM Schema Inference Architecture

```mermaid
sequenceDiagram
    participant User
    participant Inferrer as LLMSchemaInferrer
    participant LLM as OpenAI/Anthropic
    participant Rules as RuleBasedInferrer

    User->>Inferrer: infer(df)
    
    alt LLM Available
        Inferrer->>LLM: Analyze column names + samples
        LLM-->>Inferrer: Semantic types
    else Fallback
        Inferrer->>Rules: Match patterns
        Rules-->>Inferrer: Semantic types
    end
    
    Inferrer-->>User: Schema with semantic types
```

## Component Integration

```mermaid
flowchart TB
    subgraph "Data Input"
        CSV[CSV Files]
        DB[(Database)]
        API[API Endpoints]
    end

    subgraph "v1.4.0 Processing"
        CSV --> LLM[LLM Schema Inference]
        DB --> LLM
        LLM --> AM[AutoML]
        
        AM --> GEN[Generator]
        
        GEN --> DR[Drift Detection]
        DR --> |adapt| GEN
        
        GEN --> VER[Versioning]
        VER --> |commit| REPO[(Repository)]
        
        GEN --> PA[Privacy Audit]
        PA --> |pass| OUT
        PA --> |fail| GEN
    end

    subgraph "Output"
        OUT[Synthetic Data]
        REPO
    end
```

## Extension Points

### Custom Domain Generator

```python
from genesis.domains import DomainGenerator

class InsuranceGenerator(DomainGenerator):
    domain = "insurance"
    
    def generate_policies(self, n_policies: int) -> pd.DataFrame:
        # Custom implementation
        pass
    
    def generate_claims(self, n_claims: int) -> pd.DataFrame:
        # Custom implementation
        pass
```

### Custom Pipeline Node

```python
from genesis.pipeline import PipelineNode

class CustomNode(PipelineNode):
    node_type = "custom"
    
    def execute(self, inputs: dict) -> dict:
        # Custom processing
        return {"output": processed_data}
```

### Custom Privacy Attack

```python
from genesis.privacy_attacks import PrivacyAttack

class CustomAttack(PrivacyAttack):
    def run(self, real_data, synthetic_data) -> AttackResult:
        # Custom attack implementation
        return AttackResult(accuracy=0.5, risk_level="LOW")
```

## Performance Considerations

| Feature | Typical Performance | Optimization Strategy |
|---------|--------------------|-----------------------|
| AutoML | O(methods × fit_time) | Parallel method evaluation |
| Augmentation | O(n_minority × gen_time) | Batch generation |
| Privacy Attacks | O(n × m) for n records, m attacks | Sampling for large datasets |
| Drift Detection | O(n × cols) | Column-parallel computation |
| Versioning | O(1) commit, O(n) diff | Content-addressable storage |
| GPU Generation | 10-50x speedup | Batch size tuning |
| Pipeline | O(nodes) | Topological parallel execution |

## Security Considerations

1. **LLM Inference**: Never send sensitive data samples to external LLMs
2. **Privacy Attacks**: Results should be kept confidential
3. **Versioning**: Repository access should be controlled
4. **Domain Generators**: Healthcare data requires HIPAA compliance
5. **Pipeline**: Validate all inputs before execution
