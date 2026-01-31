# ADR-0020: CLI as Thin Wrapper Over Python SDK

## Status

Accepted

## Context

Genesis provides both a Python SDK and a command-line interface:

```python
# Python SDK
from genesis import SyntheticGenerator
gen = SyntheticGenerator(method='ctgan')
gen.fit(df)
synthetic = gen.generate(10000)
```

```bash
# CLI
genesis generate --input data.csv --method ctgan --samples 10000 --output synthetic.csv
```

Two architectural approaches exist for CLI implementation:

**A. CLI with independent logic**:
- CLI implements generation logic directly
- May diverge from SDK behavior
- Harder to maintain parity
- Potentially faster for simple cases

**B. CLI as thin wrapper over SDK**:
- CLI translates arguments to SDK calls
- Guaranteed behavior parity
- Single source of truth
- Slight overhead from abstraction

We prioritized consistency: users should get identical results whether using Python or CLI.

## Decision

We implement the **CLI as a thin wrapper that delegates entirely to the Python SDK**:

```python
# genesis/cli/main.py

@click.command()
@click.option('--input', '-i', required=True, help='Input CSV file')
@click.option('--output', '-o', required=True, help='Output CSV file')
@click.option('--method', '-m', default='auto', help='Generation method')
@click.option('--samples', '-n', type=int, required=True, help='Number of samples')
def generate(input, output, method, samples):
    """Generate synthetic data from a CSV file."""
    
    # 1. Load data (thin wrapper)
    data = pd.read_csv(input)
    
    # 2. Delegate to SDK (no CLI-specific logic)
    from genesis import SyntheticGenerator
    generator = SyntheticGenerator(method=method)
    generator.fit(data)
    synthetic = generator.generate(samples)
    
    # 3. Save output (thin wrapper)
    synthetic.to_csv(output, index=False)
    click.echo(f"Generated {len(synthetic)} samples to {output}")
```

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Python    │  │    CLI      │  │  REST API   │             │
│  │   Import    │  │  (Click)    │  │  (FastAPI)  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Python SDK                            │   │
│  │  SyntheticGenerator, QualityEvaluator, PrivacyConfig    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Core Engine                            │   │
│  │  Generators, Transformers, Evaluators, Constraints       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### CLI Responsibilities (Limited)

The CLI handles only:
1. **Argument parsing**: Using Click for user-friendly interface
2. **File I/O**: Reading CSVs, writing outputs
3. **Progress display**: Rich console output
4. **Error formatting**: User-friendly error messages

### CLI Does NOT Handle

- Generation logic
- Privacy calculations
- Quality evaluation
- Method selection
- Constraint validation

All of these delegate to the SDK.

## Consequences

### Positive

- **Behavior parity**: CLI and SDK produce identical results
- **Single source of truth**: Bug fixes in SDK automatically fix CLI
- **Easier testing**: SDK tests cover CLI behavior
- **Documentation sync**: SDK docs apply to CLI
- **Feature parity**: New SDK features immediately available in CLI

### Negative

- **Import overhead**: CLI loads full SDK even for simple commands
- **Abstraction cost**: Extra layer between user and core logic
- **Limited CLI optimization**: Can't bypass SDK for performance

### CLI Command Structure

```
genesis
├── generate          # Generate synthetic data
├── evaluate          # Evaluate synthetic data quality  
├── analyze           # Analyze dataset characteristics
├── automl            # Auto-select method and generate
├── augment           # Augment imbalanced datasets
├── privacy-audit     # Run privacy attack tests
├── drift             # Detect data drift
├── domain            # Domain-specific generation
│   ├── healthcare
│   ├── finance
│   └── retail
└── version           # Dataset versioning
    ├── init
    ├── commit
    ├── log
    ├── tag
    ├── diff
    └── checkout
```

## Examples

```bash
# Basic generation (maps to SyntheticGenerator)
genesis generate -i customers.csv -o synthetic.csv -n 10000

# With method selection (maps to GeneratorConfig)
genesis generate -i data.csv -o out.csv -m ctgan --epochs 500

# Evaluation (maps to QualityEvaluator)
genesis evaluate --real data.csv --synthetic synthetic.csv --output report.html

# AutoML (maps to auto_synthesize)
genesis automl -i data.csv -o synthetic.csv -n 10000

# Privacy audit (maps to run_privacy_audit)
genesis privacy-audit -r original.csv -s synthetic.csv --sensitive ssn,income

# Domain-specific (maps to DomainGenerator)
genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv
```

### Implementation Pattern

Every CLI command follows this pattern:

```python
@click.command()
@click.option('--input', '-i', required=True)
@click.option('--output', '-o', required=True)
# ... more options
def command_name(input, output, **kwargs):
    """Docstring becomes --help text."""
    
    # 1. Parse/validate arguments (CLI responsibility)
    if not Path(input).exists():
        raise click.BadParameter(f"File not found: {input}")
    
    # 2. Load data (CLI responsibility)
    data = pd.read_csv(input)
    
    # 3. Call SDK (delegation)
    from genesis import SomeSDKClass
    result = SomeSDKClass(**kwargs).process(data)
    
    # 4. Save/display output (CLI responsibility)
    result.to_csv(output)
    
    # 5. User feedback (CLI responsibility)
    console.print(f"[green]✓[/green] Processed {len(data)} rows")
```

## Testing Strategy

```python
# CLI tests verify argument parsing and I/O
def test_generate_command(tmp_path):
    input_file = tmp_path / "input.csv"
    output_file = tmp_path / "output.csv"
    
    # Create test input
    pd.DataFrame({'a': [1, 2, 3]}).to_csv(input_file)
    
    # Run CLI
    result = runner.invoke(cli, [
        'generate', '-i', str(input_file), '-o', str(output_file), '-n', '10'
    ])
    
    assert result.exit_code == 0
    assert output_file.exists()

# SDK tests verify actual generation logic
def test_synthetic_generator():
    gen = SyntheticGenerator(method='gaussian_copula')
    gen.fit(test_data)
    synthetic = gen.generate(100)
    
    assert len(synthetic) == 100
    assert list(synthetic.columns) == list(test_data.columns)
```
