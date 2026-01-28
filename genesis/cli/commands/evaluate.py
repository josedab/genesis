"""Evaluation commands for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--real", "-r", "real_path", required=True, help="Path to real data file")
@click.option(
    "--synthetic", "-s", "synthetic_path", required=True, help="Path to synthetic data file"
)
@click.option("--output", "-o", "output_path", default=None, help="Path to save report")
@click.option(
    "--format", "-f", "output_format", default="text", help="Output format (text, json, html)"
)
@click.option("--target", "-t", default=None, help="Target column for ML utility evaluation")
def evaluate(
    real_path: str,
    synthetic_path: str,
    output_path: str,
    output_format: str,
    target: str,
):
    """Evaluate synthetic data quality.

    Example:
        genesis evaluate -r data.csv -s synthetic.csv -o report.html -f html
    """
    import pandas as pd

    from genesis import QualityEvaluator

    console.print("[bold blue]Genesis[/] - Evaluating synthetic data quality")

    # Load data
    real_df = pd.read_csv(real_path) if real_path.endswith(".csv") else pd.read_parquet(real_path)
    syn_df = (
        pd.read_csv(synthetic_path)
        if synthetic_path.endswith(".csv")
        else pd.read_parquet(synthetic_path)
    )

    console.print(f"Real data: {len(real_df)} rows")
    console.print(f"Synthetic data: {len(syn_df)} rows")

    # Evaluate
    evaluator = QualityEvaluator(real_df, syn_df)
    report = evaluator.evaluate(target_column=target)

    # Output
    if output_format == "text":
        console.print(report.summary())
    elif output_format == "json":
        output = report.to_json()
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
            console.print(f"[bold green]✓[/] Report saved to {output_path}")
        else:
            console.print(output)
    elif output_format == "html":
        output = report.to_html()
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
            console.print(f"[bold green]✓[/] Report saved to {output_path}")
        else:
            console.print(output)
    else:
        console.print(report.summary())


@click.command()
@click.option("--real", "-r", "real_path", required=True, help="Path to real data file")
@click.option(
    "--synthetic", "-s", "synthetic_path", required=True, help="Path to synthetic data file"
)
@click.option("--output", "-o", "output_path", required=True, help="Path to save report")
@click.option("--target", "-t", default=None, help="Target column for ML utility")
def report(real_path: str, synthetic_path: str, output_path: str, target: str):
    """Generate comprehensive quality report.

    Example:
        genesis report -r data.csv -s synthetic.csv -o report.html
    """
    import pandas as pd

    from genesis import QualityEvaluator

    console.print("[bold blue]Genesis[/] - Generating quality report")

    # Load data
    real_df = pd.read_csv(real_path) if real_path.endswith(".csv") else pd.read_parquet(real_path)
    syn_df = (
        pd.read_csv(synthetic_path)
        if synthetic_path.endswith(".csv")
        else pd.read_parquet(synthetic_path)
    )

    # Evaluate
    evaluator = QualityEvaluator(real_df, syn_df)
    report_obj = evaluator.evaluate(target_column=target)

    # Save report
    if output_path.endswith(".html"):
        report_obj.save_html(output_path)
    else:
        report_obj.save_json(output_path)

    console.print(f"[bold green]✓[/] Report saved to {output_path}")
    console.print("\n[bold]Summary[/]")
    console.print(f"  Overall Score: {report_obj.overall_score:.1f}%")
    console.print(f"  Statistical Fidelity: {report_obj.fidelity_score * 100:.1f}%")
    console.print(f"  ML Utility: {report_obj.utility_score * 100:.1f}%")
    console.print(f"  Privacy: {report_obj.privacy_score * 100:.1f}%")


@click.command()
@click.option("--baseline", "-b", "baseline_path", required=True, help="Path to baseline data file")
@click.option("--current", "-c", "current_path", required=True, help="Path to current data file")
@click.option("--output", "-o", "output_path", default=None, help="Path to save report")
@click.option(
    "--format", "-f", "output_format", default="text", help="Output format: text, json, html"
)
@click.option("--generate", is_flag=True, help="Generate drift-adapted synthetic data")
@click.option("--samples", "-n", default=1000, type=int, help="Number of samples if generating")
def drift(
    baseline_path: str,
    current_path: str,
    output_path: str,
    output_format: str,
    generate: bool,
    samples: int,
):
    """Detect data drift between two datasets.

    Example:
        genesis drift -b baseline.csv -c current.csv
        genesis drift -b baseline.csv -c current.csv --generate -n 10000 -o adapted.csv
    """
    import pandas as pd

    from genesis.drift import DriftAwareGenerator, detect_drift

    console.print("[bold blue]Genesis Drift Detection[/]")

    # Load data
    baseline_df = (
        pd.read_csv(baseline_path)
        if baseline_path.endswith(".csv")
        else pd.read_parquet(baseline_path)
    )
    current_df = (
        pd.read_csv(current_path)
        if current_path.endswith(".csv")
        else pd.read_parquet(current_path)
    )

    console.print(f"Baseline: {len(baseline_df)} rows")
    console.print(f"Current: {len(current_df)} rows")

    # Detect drift
    console.print("\n[dim]Detecting drift...[/]")
    drift_report = detect_drift(baseline_df, current_df)

    # Display results
    console.print("\n[bold]Drift Analysis:[/]")
    console.print(f"  Overall Drift Score: {drift_report.overall_drift_score:.4f}")
    console.print(
        f"  Significant Drift: [{'red' if drift_report.has_significant_drift else 'green'}]{drift_report.has_significant_drift}[/]"
    )

    if drift_report.drifted_columns:
        console.print("\n[bold]Drifted Columns:[/]")
        for col in drift_report.drifted_columns[:10]:
            result = drift_report.column_results[col]
            console.print(f"  {col}: score={result.drift_score:.4f}")
    else:
        console.print("\n[green]No significant drift detected[/]")

    # Save report
    if output_path and not generate:
        if output_format == "html":
            drift_report.to_html(output_path)
        elif output_format == "json":
            drift_report.to_json(output_path)
        console.print(f"\n[bold green]✓[/] Report saved to {output_path}")

    # Generate adapted data
    if generate:
        console.print("\n[dim]Generating drift-adapted data...[/]")
        from genesis.generators.tabular import GaussianCopulaGenerator

        base_generator = GaussianCopulaGenerator()
        generator = DriftAwareGenerator(generator=base_generator)
        generator.fit(baseline_df)
        synthetic = generator.generate(n_samples=samples, current_data=current_df)

        out_path = output_path or "drift_adapted.csv"
        if out_path.endswith(".csv"):
            synthetic.to_csv(out_path, index=False)
        else:
            synthetic.to_parquet(out_path, index=False)
        console.print(f"[bold green]✓[/] Generated {len(synthetic)} adapted samples to {out_path}")
