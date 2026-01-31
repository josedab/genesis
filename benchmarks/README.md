# Genesis Benchmarks

This directory contains benchmarks for measuring Genesis performance.

## Running Benchmarks

```bash
# Basic run
python benchmarks/run_benchmarks.py

# Custom configuration
python benchmarks/run_benchmarks.py --sizes 1000 5000 10000 --samples 2000

# Save results
python benchmarks/run_benchmarks.py --output results.json
```

## Benchmark Metrics

- **Fit Time**: Time to train the generator on real data
- **Generate Time**: Time to generate synthetic samples
- **Samples/Second**: Generation throughput
- **Memory Usage**: Peak memory during generation
- **Fidelity Score**: Statistical similarity to real data

## Results

Results are saved as JSON for historical comparison. Example:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "python_version": "3.11.0",
  "results": [
    {
      "name": "GaussianCopulaGenerator",
      "dataset_rows": 1000,
      "fit_time_seconds": 0.5,
      "generate_time_seconds": 0.1,
      "samples_per_second": 10000.0,
      "fidelity_score": 0.92
    }
  ]
}
```

## Adding New Benchmarks

1. Add generator to `generators` list in `run_benchmarks.py`
2. Configure any special parameters
3. Run and compare results
