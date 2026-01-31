# Genesis v1.4.0 API Reference

Complete API reference for all v1.4.0 modules.

## AutoML Module

### `genesis.automl`

#### Classes

##### `MetaFeatureExtractor`

Extracts meta-features from a dataset for method selection.

```python
class MetaFeatureExtractor:
    def extract(self, data: pd.DataFrame) -> MetaFeatures:
        """Extract meta-features from dataset."""
```

**Returns**: `MetaFeatures` dataclass with:
- `n_rows: int` - Number of rows
- `n_columns: int` - Number of columns  
- `numeric_ratio: float` - Ratio of numeric columns
- `categorical_ratio: float` - Ratio of categorical columns
- `datetime_ratio: float` - Ratio of datetime columns
- `missing_ratio: float` - Ratio of missing values
- `has_high_cardinality: bool` - Has high cardinality categoricals
- `has_temporal: bool` - Has temporal columns
- `max_correlation: float` - Maximum pairwise correlation
- `memory_mb: float` - Estimated memory usage

##### `MethodSelector`

Selects optimal generation method based on meta-features.

```python
class MethodSelector:
    def __init__(
        self,
        prefer_speed: bool = False,
        prefer_quality: bool = False
    ):
        """Initialize method selector."""
    
    def select(self, features: MetaFeatures) -> SelectionResult:
        """Select best method for given features."""
```

**Returns**: `SelectionResult` with:
- `recommended_method: GenerationMethod`
- `confidence: float`
- `reason: str`
- `all_recommendations: List[MethodRecommendation]`

##### `AutoMLSynthesizer`

End-to-end automatic synthesis.

```python
class AutoMLSynthesizer:
    def __init__(
        self,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        exclude_methods: List[GenerationMethod] = None,
        force_method: GenerationMethod = None,
        max_epochs: int = 300,
        batch_size: int = 500,
        verbose: bool = True
    ):
        """Initialize AutoML synthesizer."""
    
    def fit(self, data: pd.DataFrame) -> 'AutoMLSynthesizer':
        """Fit synthesizer to data."""
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples."""
    
    @property
    def selected_method(self) -> GenerationMethod:
        """Get selected generation method."""
    
    @property
    def selection_confidence(self) -> float:
        """Get method selection confidence."""
```

#### Functions

##### `auto_synthesize`

```python
def auto_synthesize(
    data: pd.DataFrame,
    n_samples: int,
    prefer_speed: bool = False,
    prefer_quality: bool = False,
    **kwargs
) -> pd.DataFrame:
    """One-line automatic synthesis."""
```

---

## Augmentation Module

### `genesis.augmentation`

#### Classes

##### `SyntheticAugmenter`

```python
class SyntheticAugmenter:
    def __init__(
        self,
        strategy: str = "oversample",  # "oversample", "smote", "combined"
        quality_threshold: float = 0.7,
        validate_samples: bool = True
    ):
        """Initialize augmenter."""
    
    def fit(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> 'SyntheticAugmenter':
        """Fit augmenter to imbalanced data."""
    
    def augment(
        self,
        target_ratio: float = 1.0,
        target_counts: Dict[str, int] = None
    ) -> pd.DataFrame:
        """Augment data to target distribution."""
    
    def quality_report(self) -> Dict[str, Any]:
        """Get augmentation quality report."""
```

##### `AugmentationPlanner`

```python
class AugmentationPlanner:
    def analyze(
        self,
        data: pd.DataFrame,
        target_column: str
    ) -> AugmentationPlan:
        """Analyze imbalance and recommend strategy."""
```

**Returns**: `AugmentationPlan` with:
- `imbalance_ratio: float`
- `majority_class: str`
- `minority_classes: List[str]`
- `recommended_strategy: str`
- `samples_needed: Dict[str, int]`

#### Functions

##### `augment_imbalanced`

```python
def augment_imbalanced(
    data: pd.DataFrame,
    target_column: str,
    strategy: str = "oversample",
    target_ratio: float = 1.0
) -> pd.DataFrame:
    """Convenience function for data augmentation."""
```

---

## Privacy Attacks Module

### `genesis.privacy_attacks`

#### Classes

##### `MembershipInferenceAttack`

```python
class MembershipInferenceAttack:
    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        holdout_data: pd.DataFrame = None
    ) -> MembershipResult:
        """Run membership inference attack."""
```

**Returns**: `MembershipResult` with:
- `accuracy: float`
- `advantage: float`
- `risk_level: str` ("LOW", "MEDIUM", "HIGH")

##### `AttributeInferenceAttack`

```python
class AttributeInferenceAttack:
    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_column: str,
        known_columns: List[str]
    ) -> AttributeResult:
        """Run attribute inference attack."""
```

**Returns**: `AttributeResult` with:
- `accuracy: float`
- `baseline: float`
- `lift: float`
- `risk_level: str`

##### `ReidentificationAttack`

```python
class ReidentificationAttack:
    def run(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        quasi_identifiers: List[str]
    ) -> ReidentificationResult:
        """Run re-identification attack."""
```

**Returns**: `ReidentificationResult` with:
- `reidentification_rate: float`
- `unique_matches: int`
- `risk_level: str`

##### `PrivacyAttackTester`

```python
class PrivacyAttackTester:
    def __init__(
        self,
        sensitive_columns: List[str] = None,
        quasi_identifiers: List[str] = None,
        thresholds: RiskThresholds = None
    ):
        """Initialize privacy tester."""
    
    def run_all_attacks(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        holdout_data: pd.DataFrame = None
    ) -> PrivacyAuditReport:
        """Run all privacy attacks."""
```

#### Functions

##### `run_privacy_audit`

```python
def run_privacy_audit(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: List[str] = None,
    quasi_identifiers: List[str] = None,
    holdout_data: pd.DataFrame = None
) -> PrivacyAuditReport:
    """Run comprehensive privacy audit."""
```

---

## LLM Inference Module

### `genesis.llm_inference`

#### Classes

##### `LLMSchemaInferrer`

```python
class LLMSchemaInferrer:
    def __init__(
        self,
        provider: str = "openai",  # "openai" or "anthropic"
        api_key: str = None,
        model: str = "gpt-4o-mini",
        max_samples: int = 10
    ):
        """Initialize LLM schema inferrer."""
    
    def infer(self, data: pd.DataFrame) -> Dict[str, ColumnSchema]:
        """Infer column schemas using LLM."""
```

##### `RuleBasedInferrer`

```python
class RuleBasedInferrer:
    def infer(self, data: pd.DataFrame) -> Dict[str, ColumnSchema]:
        """Infer schemas using pattern matching."""
    
    def add_pattern(
        self,
        pattern: str,
        semantic_type: str,
        data_type: str
    ):
        """Add custom pattern."""
```

---

## Drift Module

### `genesis.drift`

#### Classes

##### `DataDriftDetector`

```python
class DataDriftDetector:
    def __init__(
        self,
        numeric_method: str = "ks",  # "ks", "psi", "wasserstein"
        categorical_method: str = "js",  # "js", "chi2", "psi"
        significance_level: float = 0.05
    ):
        """Initialize drift detector."""
    
    def detect(
        self,
        baseline: pd.DataFrame,
        current: pd.DataFrame
    ) -> DriftReport:
        """Detect drift between datasets."""
```

**Returns**: `DriftReport` with:
- `overall_drift_score: float`
- `has_significant_drift: bool`
- `drifted_columns: List[str]`
- `column_results: Dict[str, ColumnDriftResult]`

##### `DriftAwareGenerator`

```python
class DriftAwareGenerator:
    def fit(self, baseline: pd.DataFrame) -> 'DriftAwareGenerator':
        """Fit on baseline data."""
    
    def generate(
        self,
        n_samples: int,
        target_distribution: pd.DataFrame = None,
        drift_adaptation: str = "weighted",  # "weighted", "blend", "retrain"
        adaptation_strength: float = 0.5
    ) -> pd.DataFrame:
        """Generate drift-aware synthetic data."""
```

##### `ContinuousMonitor`

```python
class ContinuousMonitor:
    def __init__(
        self,
        baseline: pd.DataFrame,
        check_interval: str = "1h",
        alert_threshold: float = 0.1
    ):
        """Initialize continuous monitor."""
    
    def add_batch(self, batch: pd.DataFrame):
        """Add new data batch."""
    
    def has_significant_drift(self) -> bool:
        """Check if drift detected."""
    
    def on_drift(self, callback: Callable):
        """Register drift alert callback."""
```

#### Functions

##### `detect_drift`

```python
def detect_drift(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    significance_level: float = 0.05
) -> DriftReport:
    """Convenience function for drift detection."""
```

##### `calculate_psi`

```python
def calculate_psi(
    baseline: pd.DataFrame,
    current: pd.DataFrame,
    bins: int = 10
) -> Dict[str, float]:
    """Calculate Population Stability Index per column."""
```

---

## Versioning Module

### `genesis.versioning`

#### Classes

##### `DatasetRepository`

```python
class DatasetRepository:
    @classmethod
    def init(cls, path: str) -> 'DatasetRepository':
        """Initialize new repository."""
    
    def __init__(self, path: str):
        """Open existing repository."""
    
    def commit(
        self,
        data: pd.DataFrame,
        message: str,
        metadata: Dict[str, Any] = None
    ) -> Commit:
        """Commit dataset version."""
    
    def log(self, branch: str = None) -> List[Commit]:
        """Get commit history."""
    
    def get_commit(self, ref: str) -> Commit:
        """Get specific commit."""
    
    def branch(self, name: str):
        """Create new branch."""
    
    def checkout(self, ref: str):
        """Checkout branch or commit."""
    
    def merge(self, branch: str):
        """Merge branch into current."""
    
    def tag(self, name: str, message: str = None):
        """Create tag at current commit."""
    
    def tags(self) -> List[Tag]:
        """List all tags."""
    
    def branches(self) -> List[Branch]:
        """List all branches."""
    
    def diff(self, ref1: str, ref2: str) -> DatasetDiff:
        """Compare two versions."""
    
    def get_current_data(self) -> pd.DataFrame:
        """Get data at current HEAD."""
```

##### `VersionedGenerator`

```python
class VersionedGenerator:
    def __init__(
        self,
        method: str = "auto",
        repository: str = None
    ):
        """Initialize versioned generator."""
    
    def fit(self, data: pd.DataFrame):
        """Fit generator."""
    
    def generate(
        self,
        n_samples: int,
        message: str = None
    ) -> pd.DataFrame:
        """Generate and auto-commit."""
```

---

## GPU Module

### `genesis.gpu`

#### Classes

##### `BatchedGenerator`

```python
class BatchedGenerator:
    def __init__(
        self,
        method: str = "ctgan",
        device: str = "cuda",
        batch_size: int = 10000,
        memory_fraction: float = 0.8,
        mixed_precision: bool = False
    ):
        """Initialize batched generator."""
    
    def fit(self, data: pd.DataFrame):
        """Fit on GPU."""
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate in batches."""
    
    def generate_batches(
        self,
        n_samples: int,
        batch_size: int = None
    ) -> Iterator[pd.DataFrame]:
        """Generate as iterator of batches."""
```

##### `MultiGPUGenerator`

```python
class MultiGPUGenerator:
    def __init__(
        self,
        method: str = "ctgan",
        strategy: str = "data_parallel",  # "data_parallel", "model_parallel"
        devices: List[str] = None
    ):
        """Initialize multi-GPU generator."""
    
    def fit(self, data: pd.DataFrame):
        """Fit across GPUs."""
    
    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate using all GPUs."""
```

#### Functions

##### `detect_gpus`

```python
def detect_gpus() -> List[GPUInfo]:
    """Detect available GPUs."""
```

##### `optimize_batch_size`

```python
def optimize_batch_size(
    model_type: str,
    n_columns: int,
    gpu_index: int = 0
) -> int:
    """Find optimal batch size for GPU."""
```

##### `estimate_memory`

```python
def estimate_memory(
    method: str,
    n_rows: int,
    n_columns: int
) -> int:
    """Estimate GPU memory in MB."""
```

---

## Domain Generators Module

### `genesis.domains`

#### Classes

##### `HealthcareGenerator`

```python
class HealthcareGenerator:
    def __init__(self, privacy_config: Dict = None):
        """Initialize healthcare generator."""
    
    def generate_patient_cohort(
        self,
        n_patients: int,
        age_range: Tuple[int, int] = (0, 100),
        include_demographics: bool = True,
        include_conditions: bool = True
    ) -> pd.DataFrame:
        """Generate patient cohort."""
    
    def generate_clinical_events(
        self,
        patient_ids: List,
        event_types: List[str],
        time_range: Tuple[str, str]
    ) -> pd.DataFrame:
        """Generate clinical events."""
    
    def generate_lab_results(
        self,
        n_results: int,
        tests: List[str] = None
    ) -> pd.DataFrame:
        """Generate lab results."""
```

##### `FinanceGenerator`

```python
class FinanceGenerator:
    def generate_transactions(
        self,
        n_transactions: int,
        accounts: int = 100,
        include_fraud: bool = False,
        fraud_rate: float = 0.02
    ) -> pd.DataFrame:
        """Generate financial transactions."""
    
    def generate_accounts(
        self,
        n_accounts: int,
        account_types: List[str] = None
    ) -> pd.DataFrame:
        """Generate bank accounts."""
    
    def generate_credit_profiles(
        self,
        n_profiles: int,
        score_range: Tuple[int, int] = (300, 850)
    ) -> pd.DataFrame:
        """Generate credit profiles."""
```

##### `RetailGenerator`

```python
class RetailGenerator:
    def generate_customers(
        self,
        n_customers: int,
        segments: List[str] = None
    ) -> pd.DataFrame:
        """Generate retail customers."""
    
    def generate_orders(
        self,
        n_orders: int,
        customer_ids: List = None
    ) -> pd.DataFrame:
        """Generate orders."""
    
    def generate_products(
        self,
        n_products: int,
        categories: List[str] = None
    ) -> pd.DataFrame:
        """Generate product catalog."""
    
    def generate_ecommerce_dataset(
        self,
        n_customers: int,
        n_products: int,
        n_orders: int
    ) -> Dict[str, pd.DataFrame]:
        """Generate complete e-commerce dataset."""
```

---

## Pipeline Module

### `genesis.pipeline`

#### Classes

##### `PipelineBuilder`

```python
class PipelineBuilder:
    def source(
        self,
        path_or_df: Union[str, pd.DataFrame],
        name: str = None
    ) -> 'PipelineBuilder':
        """Add source node."""
    
    def transform(
        self,
        operation: str,
        config: Dict = None,
        input: str = None,
        name: str = None
    ) -> 'PipelineBuilder':
        """Add transform node."""
    
    def synthesize(
        self,
        method: str = "auto",
        n_samples: int = None,
        config: Dict = None,
        input: str = None,
        name: str = None
    ) -> 'PipelineBuilder':
        """Add synthesize node."""
    
    def evaluate(
        self,
        metrics: List[str] = None,
        output: str = None,
        input: str = None
    ) -> 'PipelineBuilder':
        """Add evaluate node."""
    
    def sink(
        self,
        path: str,
        input: str = None,
        name: str = None
    ) -> 'PipelineBuilder':
        """Add sink node."""
    
    def add_node(
        self,
        name: str,
        node_type: str,
        config: Dict = None,
        input: str = None
    ) -> 'PipelineBuilder':
        """Add custom node."""
    
    def build(self) -> Pipeline:
        """Build pipeline."""
```

##### `Pipeline`

```python
class Pipeline:
    def execute(
        self,
        config: Dict = None,
        parallel: bool = False
    ) -> Dict[str, Any]:
        """Execute pipeline."""
    
    def validate(self) -> ValidationResult:
        """Validate pipeline."""
    
    def save(self, path: str):
        """Save to YAML."""
    
    @classmethod
    def load(cls, path: str) -> 'Pipeline':
        """Load from YAML."""
```

#### Functions

##### `create_simple_pipeline`

```python
def create_simple_pipeline(
    source: str,
    output: str,
    n_samples: int,
    method: str = "auto"
) -> Pipeline:
    """Create simple generation pipeline."""
```

##### `create_evaluation_pipeline`

```python
def create_evaluation_pipeline(
    real_data: str,
    synthetic_data: str,
    report_output: str
) -> Pipeline:
    """Create evaluation pipeline."""
```
