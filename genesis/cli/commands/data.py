"""Data operations commands for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to input data file")
@click.option("--output", "-o", "output_path", required=True, help="Path to output file")
@click.option(
    "--target", "-t", "target_column", required=True, help="Target column for augmentation"
)
@click.option(
    "--strategy", "-s", default="oversample", help="Strategy: oversample, smote, combined"
)
@click.option("--ratio", "-r", default=1.0, type=float, help="Target ratio (1.0 = balanced)")
@click.option("--analyze-only", is_flag=True, help="Only analyze imbalance, don't augment")
def augment(
    input_path: str,
    output_path: str,
    target_column: str,
    strategy: str,
    ratio: float,
    analyze_only: bool,
):
    """Augment imbalanced dataset with synthetic samples.

    Example:
        genesis augment -i imbalanced.csv -o balanced.csv -t label
        genesis augment -i data.csv -t fraud --analyze-only
    """
    import pandas as pd

    from genesis.augmentation import AugmentationPlanner, SyntheticAugmenter

    console.print("[bold blue]Genesis Augmentation[/]")

    # Load data
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)
    console.print(f"Loaded: {len(df)} rows")

    # Analyze
    planner = AugmentationPlanner()
    plan = planner.analyze(df, target_column)

    console.print("\n[bold]Class Distribution:[/]")
    for cls, count in df[target_column].value_counts().items():
        pct = count / len(df) * 100
        console.print(f"  {cls}: {count} ({pct:.1f}%)")

    console.print("\n[bold]Analysis:[/]")
    console.print(f"  Imbalance ratio: {plan.imbalance_ratio:.2f}")
    console.print(f"  Majority class: {plan.majority_class}")
    console.print(f"  Recommended strategy: {plan.recommended_strategy}")

    if analyze_only:
        console.print("\n[bold]Samples needed for balance:[/]")
        for cls, needed in plan.samples_needed.items():
            console.print(f"  {cls}: {needed}")
        return

    # Augment
    console.print(f"\n[dim]Augmenting with strategy: {strategy}...[/]")

    augmenter = SyntheticAugmenter(strategy=strategy)
    augmenter.fit(df, target_column)
    balanced = augmenter.augment(target_ratio=ratio)

    # Save
    if output_path.endswith(".csv"):
        balanced.to_csv(output_path, index=False)
    else:
        balanced.to_parquet(output_path, index=False)

    console.print(f"[bold green]✓[/] Saved {len(balanced)} rows to {output_path}")
    console.print("\n[bold]New Distribution:[/]")
    for cls, count in balanced[target_column].value_counts().items():
        pct = count / len(balanced) * 100
        console.print(f"  {cls}: {count} ({pct:.1f}%)")


@click.command()
@click.argument("domain_name", type=click.Choice(["healthcare", "finance", "retail"]))
@click.option("--type", "-t", "data_type", required=True, help="Data type to generate")
@click.option("--samples", "-n", default=1000, type=int, help="Number of samples")
@click.option("--output", "-o", "output_path", required=True, help="Output file path")
@click.option("--fraud-rate", default=0.02, type=float, help="Fraud rate for finance data")
def domain(
    domain_name: str, data_type: str, samples: int, output_path: str, fraud_rate: float
) -> None:
    """Generate domain-specific synthetic data.

    Example:
        genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv
        genesis domain finance -t transactions -n 10000 --fraud-rate 0.02 -o txns.csv
        genesis domain retail -t orders -n 5000 -o orders.csv
    """
    from genesis.domains import FinanceGenerator, HealthcareGenerator, RetailGenerator

    console.print(f"[bold blue]Genesis Domain Generator[/] - {domain_name.title()}")

    df = None
    if domain_name == "healthcare":
        healthcare_gen = HealthcareGenerator()
        if data_type == "patient_cohort":
            df = healthcare_gen.generate_patient_cohort(n_patients=samples)
        elif data_type == "lab_results":
            df = healthcare_gen.generate_lab_results(n_results=samples)
        else:
            console.print(f"[red]Unknown type: {data_type}. Try: patient_cohort, lab_results[/]")
            return

    elif domain_name == "finance":
        finance_gen = FinanceGenerator()
        if data_type == "transactions":
            df = finance_gen.generate_transactions(
                n_transactions=samples, include_fraud=True, fraud_rate=fraud_rate
            )
        elif data_type == "accounts":
            df = finance_gen.generate_accounts(n_accounts=samples)
        else:
            console.print(f"[red]Unknown type: {data_type}. Try: transactions, accounts[/]")
            return

    elif domain_name == "retail":
        retail_gen = RetailGenerator()
        if data_type == "customers":
            df = retail_gen.generate_customers(n_customers=samples)
        elif data_type == "orders":
            df = retail_gen.generate_orders(n_orders=samples)
        elif data_type == "products":
            df = retail_gen.generate_products(n_products=samples)
        else:
            console.print(f"[red]Unknown type: {data_type}. Try: customers, orders, products[/]")
            return

    if df is None:
        console.print(f"[red]Unknown domain: {domain_name}[/]")
        return

    # Save
    if output_path.endswith(".csv"):
        df.to_csv(output_path, index=False)
    else:
        df.to_parquet(output_path, index=False)

    console.print(f"[bold green]✓[/] Generated {len(df)} {data_type} records to {output_path}")


@click.command()
@click.option("--config", "-c", "config_path", required=True, help="Pipeline YAML config file")
@click.option("--validate-only", is_flag=True, help="Only validate, don't execute")
def pipeline(config_path: str, validate_only: bool):
    """Execute a data generation pipeline.

    Example:
        genesis pipeline -c pipeline.yaml
        genesis pipeline -c pipeline.yaml --validate-only
    """
    from genesis.pipeline import Pipeline

    console.print("[bold blue]Genesis Pipeline[/]")
    console.print(f"Loading: {config_path}")

    pipeline_obj = Pipeline.load(config_path)

    # Validate
    validation = pipeline_obj.validate()
    if not validation.is_valid:
        console.print("[red]Validation errors:[/]")
        for error in validation.errors:
            console.print(f"  - {error}")
        return

    console.print(f"[green]✓[/] Pipeline validated: {len(pipeline_obj.nodes)} nodes")

    if validate_only:
        return

    # Execute
    console.print("\n[dim]Executing pipeline...[/]")
    result = pipeline_obj.execute()

    console.print("\n[bold green]✓[/] Pipeline completed")
    for node_name, node_result in result.items():
        if isinstance(node_result, dict) and "n_rows" in node_result:
            console.print(f"  {node_name}: {node_result['n_rows']} rows")
