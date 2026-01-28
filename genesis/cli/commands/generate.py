"""Generate command for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to input data file")
@click.option("--output", "-o", "output_path", required=True, help="Path to output file")
@click.option(
    "--method", "-m", default="auto", help="Generation method (auto, ctgan, tvae, gaussian_copula)"
)
@click.option("--samples", "-n", default=None, type=int, help="Number of samples to generate")
@click.option(
    "--epochs", "-e", default=300, type=int, help="Training epochs (for deep learning methods)"
)
@click.option(
    "--discrete", "-d", multiple=True, help="Discrete column names (can specify multiple)"
)
@click.option("--privacy/--no-privacy", default=False, help="Enable differential privacy")
@click.option("--epsilon", default=1.0, type=float, help="Privacy budget (lower = more private)")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
@click.option("--verbose/--quiet", default=True, help="Print progress")
def generate(
    input_path: str,
    output_path: str,
    method: str,
    samples: int,
    epochs: int,
    discrete: tuple,
    privacy: bool,
    epsilon: float,
    seed: int,
    verbose: bool,
):
    """Generate synthetic data from input file.

    Example:
        genesis generate -i data.csv -o synthetic.csv -n 1000 --method ctgan
    """
    import pandas as pd

    from genesis import GeneratorConfig, PrivacyConfig, SyntheticGenerator

    if verbose:
        console.print(f"[bold blue]Genesis[/] - Loading data from {input_path}")

    # Load data
    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)

    # Determine sample size
    n_samples = samples if samples else len(df)

    if verbose:
        console.print(f"Input: {len(df)} rows, {len(df.columns)} columns")
        console.print(f"Generating: {n_samples} synthetic samples")

    # Configure
    config = GeneratorConfig(
        method=method,
        epochs=epochs,
        random_seed=seed,
        verbose=verbose,
    )

    privacy_config = None
    if privacy:
        privacy_config = PrivacyConfig(
            enable_differential_privacy=True,
            epsilon=epsilon,
        )

    # Create generator
    generator = SyntheticGenerator(method=method, config=config, privacy=privacy_config)

    # Fit and generate
    discrete_cols = list(discrete) if discrete else None
    generator.fit(df, discrete_columns=discrete_cols)
    synthetic_df = generator.generate(n_samples)

    # Save
    if output_path.endswith(".csv"):
        synthetic_df.to_csv(output_path, index=False)
    else:
        synthetic_df.to_parquet(output_path, index=False)

    if verbose:
        console.print(f"[bold green]✓[/] Saved {len(synthetic_df)} rows to {output_path}")
