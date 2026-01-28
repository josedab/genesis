"""Analysis commands for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to input data file")
@click.option("--output", "-o", "output_path", default=None, help="Path to save analysis")
@click.option("--format", "-f", "output_format", default="text", help="Output format (text, json)")
def analyze(input_path: str, output_path: str, output_format: str):
    """Analyze input data schema and statistics.

    Example:
        genesis analyze -i data.csv -o analysis.json -f json
    """
    import json

    import pandas as pd

    from genesis.analyzers import PrivacyAnalyzer, SchemaAnalyzer, StatisticalAnalyzer

    console.print(f"[bold blue]Genesis[/] - Analyzing data from {input_path}")

    # Load data
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)

    console.print(f"Data: {len(df)} rows, {len(df.columns)} columns")

    # Schema analysis
    schema_analyzer = SchemaAnalyzer()
    schema = schema_analyzer.analyze(df)

    # Statistical analysis
    stat_analyzer = StatisticalAnalyzer()
    stats = stat_analyzer.analyze(df)

    # Privacy analysis
    privacy_analyzer = PrivacyAnalyzer()
    privacy = privacy_analyzer.analyze(df)

    # Compile results
    analysis = {
        "schema": schema.to_dict(),
        "statistics": stats.to_dict(),
        "privacy": privacy.to_dict(),
    }

    if output_format == "json":
        output = json.dumps(analysis, indent=2, default=str)
        if output_path:
            with open(output_path, "w") as f:
                f.write(output)
            console.print(f"[bold green]✓[/] Analysis saved to {output_path}")
        else:
            console.print(output)
    else:
        console.print("\n[bold]Schema Summary[/]")
        console.print(f"  Rows: {schema.n_rows}")
        console.print(f"  Columns: {schema.n_columns}")
        console.print(f"  Primary Key: {schema.primary_key}")

        console.print("\n[bold]Column Types[/]")
        type_summary = schema_analyzer.get_column_types_summary(schema)
        for dtype, cols in type_summary.items():
            console.print(f"  {dtype}: {len(cols)} columns")

        console.print("\n[bold]Privacy Risk[/]")
        console.print(f"  Overall Risk Score: {privacy.overall_risk_score:.2f}")
        console.print(f"  K-Anonymity Estimate: {privacy.k_anonymity_estimate}")
        console.print(f"  Quasi-Identifiers: {privacy.quasi_identifiers[:5]}...")


@click.command("privacy-audit")
@click.option("--real", "-r", "real_path", required=True, help="Path to real data file")
@click.option(
    "--synthetic", "-s", "synthetic_path", required=True, help="Path to synthetic data file"
)
@click.option("--holdout", "-h", "holdout_path", default=None, help="Path to holdout data file")
@click.option("--sensitive", default=None, help="Comma-separated sensitive columns")
@click.option("--quasi-ids", default=None, help="Comma-separated quasi-identifier columns")
@click.option("--output", "-o", "output_path", default=None, help="Path to save report")
@click.option(
    "--attack", default=None, help="Specific attack: membership, attribute, reidentification"
)
def privacy_audit(
    real_path: str,
    synthetic_path: str,
    holdout_path: str,
    sensitive: str,
    quasi_ids: str,
    output_path: str,
    attack: str,
):
    """Run privacy attack tests on synthetic data.

    Example:
        genesis privacy-audit -r original.csv -s synthetic.csv --sensitive ssn,income
        genesis privacy-audit -r original.csv -s synthetic.csv -o report.html
    """
    import pandas as pd

    from genesis.privacy_attacks import run_privacy_audit

    console.print("[bold blue]Genesis Privacy Audit[/]")

    # Load data
    real_df = pd.read_csv(real_path) if real_path.endswith(".csv") else pd.read_parquet(real_path)
    syn_df = (
        pd.read_csv(synthetic_path)
        if synthetic_path.endswith(".csv")
        else pd.read_parquet(synthetic_path)
    )
    holdout_df = None
    if holdout_path:
        holdout_df = (
            pd.read_csv(holdout_path)
            if holdout_path.endswith(".csv")
            else pd.read_parquet(holdout_path)
        )

    console.print(f"Real data: {len(real_df)} rows")
    console.print(f"Synthetic data: {len(syn_df)} rows")

    # Parse columns
    sensitive_cols = sensitive.split(",") if sensitive else None
    quasi_id_cols = quasi_ids.split(",") if quasi_ids else None

    # Run audit
    console.print("\n[dim]Running privacy attacks...[/]")

    audit_report = run_privacy_audit(
        real_data=real_df,
        synthetic_data=syn_df,
        sensitive_columns=sensitive_cols,
        quasi_identifiers=quasi_id_cols,
        holdout_data=holdout_df,
    )

    # Display results
    console.print("\n[bold]Privacy Audit Results:[/]")
    console.print(
        f"  Overall Risk: [{'red' if audit_report.overall_risk == 'HIGH' else 'yellow' if audit_report.overall_risk == 'MEDIUM' else 'green'}]{audit_report.overall_risk}[/]"
    )
    console.print(f"  Passed: [{'green' if audit_report.passed else 'red'}]{audit_report.passed}[/]")

    if audit_report.membership_result:
        console.print("\n  [bold]Membership Inference:[/]")
        console.print(f"    Accuracy: {audit_report.membership_result.accuracy:.1%}")
        console.print(f"    Risk: {audit_report.membership_result.risk_level}")

    if audit_report.reidentification_result:
        console.print("\n  [bold]Re-identification:[/]")
        console.print(f"    Rate: {audit_report.reidentification_result.reidentification_rate:.1%}")
        console.print(f"    Risk: {audit_report.reidentification_result.risk_level}")

    # Save report
    if output_path:
        if output_path.endswith(".html"):
            audit_report.to_html(output_path)
        else:
            audit_report.to_json(output_path)
        console.print(f"\n[bold green]✓[/] Report saved to {output_path}")
