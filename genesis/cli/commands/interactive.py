"""Interactive commands for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--data", "-d", "data_path", default=None, help="Path to reference data file")
@click.option(
    "--output", "-o", "output_path", default="synthetic_output.csv", help="Output file path"
)
@click.option("--model", "-m", default="gpt-4o-mini", help="LLM model to use")
@click.option("--api-key", envvar="OPENAI_API_KEY", help="API key (or set OPENAI_API_KEY)")
def chat(data_path: str, output_path: str, model: str, api_key: str):
    """Interactive chat interface for synthetic data generation.

    Use natural language to describe the data you want to generate.

    Example:
        genesis chat -d customers.csv -o synthetic.csv
        genesis chat --model gpt-4o
    """
    import os

    import pandas as pd
    from rich.panel import Panel

    console.print(
        Panel.fit(
            "[bold blue]Genesis Chat[/] - Natural Language Synthetic Data Generation\n"
            "Type your data generation request in plain English.\n"
            "Type 'generate' to create the data, 'quit' to exit.",
            title="Welcome",
            border_style="blue",
        )
    )

    # Load reference data if provided
    base_data = None
    if data_path:
        try:
            base_data = (
                pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
            )
            console.print(
                f"[dim]Loaded reference data: {len(base_data)} rows, {len(base_data.columns)} columns[/]"
            )
            console.print(
                f"[dim]Columns: {', '.join(base_data.columns[:10])}{'...' if len(base_data.columns) > 10 else ''}[/]"
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load reference data: {e}[/]")

    # Check for API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: No API key found. Set OPENAI_API_KEY or use --api-key[/]")
        return

    from genesis.agents import SyntheticDataAgent

    agent = SyntheticDataAgent(
        api_key=api_key,
        model=model,
        base_data=base_data,
    )

    response = None

    while True:
        try:
            user_input = console.input("\n[bold green]You>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/]")
            break

        if user_input.lower() == "generate":
            if response is None or response.config is None:
                console.print("[yellow]No configuration ready. Describe your data first.[/]")
                continue

            if response.needs_clarification:
                console.print(f"[yellow]Please answer: {response.clarification_question}[/]")
                continue

            console.print("[dim]Generating synthetic data...[/]")
            try:
                data = response.generate()

                # Save data
                if output_path.endswith(".csv"):
                    data.to_csv(output_path, index=False)
                else:
                    data.to_parquet(output_path, index=False)

                console.print(
                    f"[bold green]✓[/] Generated {len(data)} rows, saved to {output_path}"
                )
                console.print(f"\n[dim]Preview:[/]\n{data.head()}")
            except Exception as e:
                console.print(f"[red]Error generating data: {e}[/]")
            continue

        if user_input.lower() == "config":
            if response and response.config:
                console.print("[dim]Current config:[/]")
                console.print(f"  Method: {response.config.generator_method}")
                console.print(f"  Samples: {response.config.n_samples}")
                console.print(f"  Columns: {len(response.config.columns)}")
            else:
                console.print("[dim]No configuration yet.[/]")
            continue

        # Process as generation request or clarification response
        try:
            if response is None or not response.needs_clarification:
                # New request
                response = agent.process(user_input)
            else:
                # Clarification response
                response = agent.clarify(user_input)

            if response.needs_clarification:
                console.print(f"[cyan]Question:[/] {response.clarification_question}")
            else:
                console.print(f"[cyan]Agent:[/] {response.message}")
                if response.config:
                    console.print(
                        f"\n[dim]Ready to generate {response.config.n_samples} samples with {len(response.config.columns)} columns.[/]"
                    )
                    console.print("[dim]Type 'generate' to create the data.[/]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")


@click.command()
@click.option("--real", "-r", "real_path", required=True, help="Path to real data file")
@click.option(
    "--synthetic", "-s", "synthetic_path", required=True, help="Path to synthetic data file"
)
@click.option("--output", "-o", "output_path", default=None, help="Save static HTML report to path")
@click.option("--port", "-p", default=8050, type=int, help="Port for dashboard server")
@click.option("--no-browser", is_flag=True, help="Don't open browser automatically")
def dashboard(real_path: str, synthetic_path: str, output_path: str, port: int, no_browser: bool):
    """Launch interactive quality dashboard.

    Example:
        genesis dashboard -r data.csv -s synthetic.csv
        genesis dashboard -r data.csv -s synthetic.csv -o report.html
    """
    import pandas as pd

    from genesis.dashboard import QualityDashboard

    console.print("[bold blue]Genesis[/] - Quality Dashboard")

    # Load data
    real_df = pd.read_csv(real_path) if real_path.endswith(".csv") else pd.read_parquet(real_path)
    syn_df = (
        pd.read_csv(synthetic_path)
        if synthetic_path.endswith(".csv")
        else pd.read_parquet(synthetic_path)
    )

    console.print(f"Real data: {len(real_df)} rows, {len(real_df.columns)} columns")
    console.print(f"Synthetic data: {len(syn_df)} rows")

    dashboard_obj = QualityDashboard(real_df, syn_df)

    if output_path:
        # Save static report
        dashboard_obj.save_report(output_path)
        console.print(f"[bold green]✓[/] Report saved to {output_path}")
    else:
        # Run interactive server
        console.print(f"[dim]Starting dashboard on http://127.0.0.1:{port}[/]")
        dashboard_obj.run(port=port, open_browser=not no_browser)
