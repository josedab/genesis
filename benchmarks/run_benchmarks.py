"""Benchmark suite for Genesis synthetic data generation."""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    method: str
    dataset_rows: int
    dataset_cols: int
    fit_time_seconds: float
    generate_time_seconds: float
    samples_generated: int
    samples_per_second: float
    memory_mb: Optional[float] = None
    fidelity_score: Optional[float] = None
    error: Optional[str] = None


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def create_synthetic_dataset(n_rows: int, n_numeric: int, n_categorical: int) -> pd.DataFrame:
    """Create a synthetic dataset for benchmarking."""
    np.random.seed(42)
    
    data = {}
    
    # Numeric columns with different distributions
    for i in range(n_numeric):
        if i % 3 == 0:
            data[f"numeric_{i}"] = np.random.normal(50, 15, n_rows)
        elif i % 3 == 1:
            data[f"numeric_{i}"] = np.random.exponential(10, n_rows)
        else:
            data[f"numeric_{i}"] = np.random.uniform(0, 100, n_rows)
    
    # Categorical columns
    for i in range(n_categorical):
        n_categories = 3 + (i % 5)  # 3-7 categories
        categories = [f"cat_{i}_{j}" for j in range(n_categories)]
        data[f"categorical_{i}"] = np.random.choice(categories, n_rows)
    
    return pd.DataFrame(data)


def benchmark_generator(
    generator_class,
    data: pd.DataFrame,
    discrete_columns: List[str],
    n_samples: int,
    **kwargs,
) -> BenchmarkResult:
    """Benchmark a single generator."""
    name = generator_class.__name__
    method = kwargs.get("method", name)
    
    try:
        # Fit
        mem_before = get_memory_usage()
        start_fit = time.perf_counter()
        
        generator = generator_class(verbose=False, **kwargs)
        generator.fit(data, discrete_columns=discrete_columns)
        
        fit_time = time.perf_counter() - start_fit
        
        # Generate
        start_gen = time.perf_counter()
        synthetic = generator.generate(n_samples=n_samples)
        generate_time = time.perf_counter() - start_gen
        
        mem_after = get_memory_usage()
        
        # Calculate metrics
        samples_per_second = n_samples / generate_time if generate_time > 0 else 0
        
        # Calculate fidelity (optional)
        fidelity = None
        try:
            from genesis.evaluation.evaluator import QualityEvaluator
            evaluator = QualityEvaluator(data, synthetic)
            report = evaluator.evaluate()
            fidelity = report.fidelity_score
        except Exception:
            pass
        
        return BenchmarkResult(
            name=name,
            method=method,
            dataset_rows=len(data),
            dataset_cols=len(data.columns),
            fit_time_seconds=round(fit_time, 3),
            generate_time_seconds=round(generate_time, 3),
            samples_generated=n_samples,
            samples_per_second=round(samples_per_second, 1),
            memory_mb=round(mem_after - mem_before, 1) if mem_before else None,
            fidelity_score=round(fidelity, 3) if fidelity else None,
        )
        
    except Exception as e:
        return BenchmarkResult(
            name=name,
            method=method,
            dataset_rows=len(data),
            dataset_cols=len(data.columns),
            fit_time_seconds=0,
            generate_time_seconds=0,
            samples_generated=0,
            samples_per_second=0,
            error=str(e),
        )


def run_benchmarks(
    sizes: List[int] = [1000, 5000, 10000],
    n_samples: int = 1000,
) -> List[BenchmarkResult]:
    """Run all benchmarks."""
    from genesis.generators.tabular import (
        CTGANGenerator,
        GaussianCopulaGenerator,
        TVAEGenerator,
    )
    
    results = []
    
    generators = [
        (GaussianCopulaGenerator, {}),
        (CTGANGenerator, {"epochs": 50}),  # Reduced for benchmarking
        (TVAEGenerator, {"epochs": 50}),
    ]
    
    for size in sizes:
        print(f"\n{'='*60}")
        print(f"Dataset size: {size} rows")
        print('='*60)
        
        # Create dataset
        data = create_synthetic_dataset(
            n_rows=size,
            n_numeric=5,
            n_categorical=3,
        )
        discrete_cols = [c for c in data.columns if c.startswith("categorical_")]
        
        for gen_class, kwargs in generators:
            print(f"\nBenchmarking {gen_class.__name__}...")
            result = benchmark_generator(
                gen_class,
                data,
                discrete_cols,
                n_samples,
                **kwargs,
            )
            results.append(result)
            
            if result.error:
                print(f"  ERROR: {result.error}")
            else:
                print(f"  Fit: {result.fit_time_seconds}s")
                print(f"  Generate: {result.generate_time_seconds}s ({result.samples_per_second} samples/s)")
                if result.fidelity_score:
                    print(f"  Fidelity: {result.fidelity_score:.1%}")
    
    return results


def save_results(results: List[BenchmarkResult], output_path: str) -> None:
    """Save benchmark results to JSON."""
    output = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "results": [asdict(r) for r in results],
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print summary table of results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Group by method
    by_method: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.error:
            continue
        if r.name not in by_method:
            by_method[r.name] = []
        by_method[r.name].append(r)
    
    print(f"\n{'Method':<25} {'Rows':<8} {'Fit(s)':<10} {'Gen(s)':<10} {'Samples/s':<12} {'Fidelity':<10}")
    print("-"*80)
    
    for method, method_results in by_method.items():
        for r in method_results:
            fidelity_str = f"{r.fidelity_score:.1%}" if r.fidelity_score else "N/A"
            print(f"{method:<25} {r.dataset_rows:<8} {r.fit_time_seconds:<10.3f} {r.generate_time_seconds:<10.3f} {r.samples_per_second:<12.1f} {fidelity_str:<10}")


def main():
    parser = argparse.ArgumentParser(description="Run Genesis benchmarks")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1000, 5000],
        help="Dataset sizes to benchmark",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output file for results",
    )
    
    args = parser.parse_args()
    
    print("Genesis Benchmark Suite")
    print("="*60)
    print(f"Dataset sizes: {args.sizes}")
    print(f"Samples to generate: {args.samples}")
    
    results = run_benchmarks(sizes=args.sizes, n_samples=args.samples)
    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
