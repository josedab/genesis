"""AutoML commands for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to input data file")
@click.option("--output", "-o", "output_path", default=None, help="Path to output file")
@click.option("--samples", "-n", default=None, type=int, help="Number of samples to generate")
@click.option("--prefer-speed", is_flag=True, help="Prefer faster methods")
@click.option("--prefer-quality", is_flag=True, help="Prefer higher quality methods")
@click.option("--recommend-only", is_flag=True, help="Only show recommendation, don't generate")
def automl(
    input_path: str,
    output_path: str,
    samples: int,
    prefer_speed: bool,
    prefer_quality: bool,
    recommend_only: bool,
):
    """Auto-select best generation method and generate synthetic data.

    Example:
        genesis automl -i data.csv -o synthetic.csv -n 1000
        genesis automl -i data.csv --recommend-only
    """
    import pandas as pd

    from genesis.automl import AutoMLSynthesizer, MetaFeatureExtractor, MethodSelector

    console.print("[bold blue]Genesis AutoML[/] - Automatic Method Selection")

    # Load data
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    console.print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Extract features and select method
    extractor = MetaFeatureExtractor()
    features = extractor.extract(df)

    selector = MethodSelector(prefer_speed=prefer_speed, prefer_quality=prefer_quality)
    result = selector.select(features)

    console.print("\n[bold]Recommendation:[/]")
    console.print(f"  Method: [green]{result.recommended_method.value}[/]")
    console.print(f"  Confidence: {result.confidence:.1%}")
    console.print(f"  Reason: {result.reason}")

    if recommend_only:
        console.print("\n[bold]All recommendations:[/]")
        for rec in result.all_recommendations[:5]:
            console.print(f"  {rec.method.value}: {rec.confidence:.1%}")
        return

    if not output_path:
        console.print("[yellow]No output path specified. Use -o to save results.[/]")
        return

    # Generate
    n_samples = samples or len(df)
    console.print(f"\n[dim]Generating {n_samples} samples...[/]")

    automl_gen = AutoMLSynthesizer(prefer_speed=prefer_speed, prefer_quality=prefer_quality)
    automl_gen.fit(df)
    synthetic = automl_gen.generate(n_samples)

    # Save
    if output_path.endswith(".csv"):
        synthetic.to_csv(output_path, index=False)
    else:
        synthetic.to_parquet(output_path, index=False)

    console.print(f"[bold green]✓[/] Saved {len(synthetic)} rows to {output_path}")


@click.command()
@click.option("--connection", "-c", "conn_string", required=True, help="Database connection string")
@click.option("--output", "-o", "output_dir", default="./synthetic_db", help="Output directory")
@click.option("--samples", "-n", default=1000, type=int, help="Samples per table")
@click.option("--method", "-m", default="auto", help="Generation method")
def discover(conn_string: str, output_dir: str, samples: int, method: str):
    """Discover database schema and generate synthetic replica.

    Example:
        genesis discover -c "postgresql://user:pass@host/db" -o ./synthetic
    """
    from pathlib import Path

    from genesis.discovery import SchemaDiscovery

    console.print("[bold blue]Genesis[/] - Schema Discovery")
    console.print("[dim]Connecting to database...[/]")

    try:
        discovery = SchemaDiscovery.from_database(conn_string)
        schema = discovery.discover()

        console.print(f"[bold green]✓[/] Discovered {len(schema.tables)} tables")
        for table in schema.tables:
            console.print(f"  • {table.name}: {len(table.columns)} columns")

        # Create generator
        console.print("\n[dim]Training generators...[/]")
        generator = discovery.create_generator(method=method)

        # Get sample data for training
        data = {t.name: discovery.get_sample_data(t.name) for t in schema.tables}
        data = {k: v for k, v in data.items() if v is not None}

        generator.fit(data)

        # Generate
        console.print(f"[dim]Generating {samples} samples per table...[/]")
        synthetic = generator.generate(samples)

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for table_name, df in synthetic.items():
            file_path = output_path / f"{table_name}.csv"
            df.to_csv(file_path, index=False)
            console.print(f"[bold green]✓[/] Saved {table_name}: {len(df)} rows")

        console.print(f"\n[bold green]✓[/] Synthetic database saved to {output_dir}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/]")
        raise
