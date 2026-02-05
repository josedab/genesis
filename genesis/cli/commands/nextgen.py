"""Next-Gen CLI commands for Genesis v2.1.0.

Commands for real-time API, CI/CD, cloud deployment, DP queries,
leaderboard, and LLM fine-tuning features.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
@click.option("--model-dir", default="./models", help="Directory with saved models")
@click.option("--cache-size", default=1000, type=int, help="Cache size in MB")
@click.option("--workers", default=4, type=int, help="Number of workers")
def realtime(host: str, port: int, model_dir: str, cache_size: int, workers: int):
    """Start a real-time synthetic data API server.

    Example:
        genesis realtime --port 8000 --model-dir ./models
    """
    try:
        from genesis.realtime_api import RealtimeConfig, create_realtime_app

        config = RealtimeConfig(
            host=host,
            port=port,
            model_dir=model_dir,
            cache_size_mb=cache_size,
            workers=workers,
        )

        console.print(f"[bold green]Starting Genesis Real-Time API[/]")
        console.print(f"Host: {host}:{port}")
        console.print(f"Model directory: {model_dir}")
        console.print(f"Cache size: {cache_size} MB")

        app = create_realtime_app(config)

        import uvicorn

        uvicorn.run(app, host=host, port=port, workers=workers)

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/]")
        console.print("Install with: pip install fastapi uvicorn")
        sys.exit(1)


@click.command()
@click.option("--provider", type=click.Choice(["github", "gitlab"]), default="github")
@click.option("--schema-file", type=click.Path(exists=True), help="Schema file")
@click.option("--output", "-o", default=".", help="Output directory")
@click.option("--samples", default=10000, type=int, help="Samples per run")
def cicd(provider: str, schema_file: Optional[str], output: str, samples: int):
    """Generate CI/CD workflow for synthetic data pipelines.

    Example:
        genesis cicd --provider github --schema-file schema.json -o .github/workflows
    """
    from genesis.cicd import CIConfig, generate_github_workflow, generate_gitlab_ci

    config = CIConfig(
        ci_provider=provider,
        schema_path=schema_file,
        output_samples=samples,
    )

    if provider == "github":
        workflow = generate_github_workflow(config)
        output_path = Path(output) / "synthetic-data.yml"
    else:
        workflow = generate_gitlab_ci(config)
        output_path = Path(output) / ".gitlab-ci.yml"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(workflow)

    console.print(f"[green]Generated {provider.upper()} CI workflow[/]")
    console.print(f"Output: {output_path}")


@click.command()
@click.argument("provider", type=click.Choice(["aws", "gcp", "azure", "local"]))
@click.option("--project-name", default="genesis", help="Project name")
@click.option("--region", default="us-east-1", help="Cloud region")
@click.option("--output", "-o", default="./infra", help="Output directory")
@click.option("--gpu/--no-gpu", default=False, help="Enable GPU support")
@click.option("--redis/--no-redis", default=False, help="Enable Redis")
@click.option("--apply/--no-apply", default=False, help="Apply immediately")
def deploy(
    provider: str,
    project_name: str,
    region: str,
    output: str,
    gpu: bool,
    redis: bool,
    apply: bool,
):
    """Deploy Genesis to cloud or local environment.

    Example:
        genesis deploy aws --project-name my-genesis --region us-west-2
        genesis deploy local --redis
    """
    from genesis.cloud_deploy import CloudConfig, CloudDeployer, CloudProvider

    config = CloudConfig(
        provider=CloudProvider(provider),
        project_name=project_name,
        region=region,
        enable_gpu=gpu,
        redis_enabled=redis,
    )

    deployer = CloudDeployer(provider, config)
    infra_code = deployer.generate_infrastructure()

    # Write output
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if provider == "local":
        output_file = output_dir / "docker-compose.yml"
    else:
        output_file = output_dir / f"main.tf"

    output_file.write_text(infra_code)

    console.print(f"[green]Generated {provider.upper()} deployment[/]")
    console.print(f"Output: {output_file}")

    cost = deployer.estimate_cost()
    console.print(f"Estimated hourly cost: ${cost:.2f}")

    if apply:
        console.print("[yellow]Applying deployment...[/]")
        result = deployer.deploy()
        if result.success:
            console.print(f"[green]Deployed successfully![/]")
            console.print(f"API URL: {result.outputs.get('api_url', 'N/A')}")
        else:
            console.print(f"[red]Deployment failed: {result.error}[/]")
            sys.exit(1)


@click.command("dp-query")
@click.argument("query")
@click.option("--data-file", "-d", type=click.Path(exists=True), help="Input data file")
@click.option("--epsilon", default=1.0, type=float, help="Privacy budget")
@click.option("--delta", default=1e-5, type=float, help="DP delta")
@click.option("--output", "-o", help="Output file")
def dp_query_cmd(
    query: str,
    data_file: Optional[str],
    epsilon: float,
    delta: float,
    output: Optional[str],
):
    """Execute a differentially private query.

    Example:
        genesis dp-query "SELECT AVG(age) FROM users" -d data.csv --epsilon 0.5
    """
    import pandas as pd

    from genesis.dp_compiler import DPBudgetManager, DPCompiler

    # Load data
    if data_file:
        if data_file.endswith(".csv"):
            data = pd.read_csv(data_file)
        elif data_file.endswith(".parquet"):
            data = pd.read_parquet(data_file)
        else:
            data = pd.read_json(data_file)
    else:
        console.print("[red]Data file required for query execution[/]")
        sys.exit(1)

    # Create compiler
    budget = DPBudgetManager(total_epsilon=epsilon, total_delta=delta)
    compiler = DPCompiler(budget_manager=budget)

    # Execute query
    try:
        result = compiler.execute(query, {"users": data})

        console.print(f"[green]Query executed with ε={epsilon}, δ={delta}[/]")
        console.print(f"\nResult:")
        console.print(result)

        if output:
            result.to_csv(output, index=False)
            console.print(f"\nSaved to: {output}")

    except Exception as e:
        console.print(f"[red]Query failed: {e}[/]")
        sys.exit(1)


@click.command()
@click.option("--name", required=True, help="Leaderboard name")
@click.option("--submit", type=click.Path(exists=True), help="Submit results file")
@click.option("--view/--no-view", default=True, help="View leaderboard")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def leaderboard(name: str, submit: Optional[str], view: bool, fmt: str):
    """Interact with synthetic data leaderboards.

    Example:
        genesis leaderboard --name adult-census --view
        genesis leaderboard --name adult-census --submit results.json
    """
    from genesis.leaderboard import Leaderboard, Submission

    board = Leaderboard(name=name)

    if submit:
        with open(submit) as f:
            data = json.load(f)

        submission = Submission(
            submitter=data.get("submitter", "anonymous"),
            model_name=data.get("model_name", "unknown"),
            metrics=data.get("metrics", {}),
        )
        entry = board.submit(submission)
        console.print(f"[green]Submitted to leaderboard![/]")
        console.print(f"Rank: #{entry.rank}")

    if view:
        entries = board.get_entries()

        if fmt == "json":
            console.print(json.dumps([e.__dict__ for e in entries], indent=2, default=str))
        else:
            table = Table(title=f"Leaderboard: {name}")
            table.add_column("Rank", style="cyan")
            table.add_column("Submitter")
            table.add_column("Model")
            table.add_column("Score", style="green")
            table.add_column("Date")

            for entry in entries[:20]:
                table.add_row(
                    str(entry.rank),
                    entry.submitter,
                    entry.model_name,
                    f"{entry.score:.4f}",
                    entry.submitted_at.strftime("%Y-%m-%d"),
                )

            console.print(table)


@click.command("finetune-data")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output file")
@click.option("--samples", "-n", default=10000, type=int, help="Number of samples")
@click.option("--epsilon", default=1.0, type=float, help="Privacy budget")
@click.option("--paraphrase/--no-paraphrase", default=True, help="Enable paraphrasing")
@click.option("--scrub-pii/--no-scrub-pii", default=True, help="Remove PII")
@click.option("--audit/--no-audit", default=True, help="Run memorization audit")
def finetune_data(
    input_file: str,
    output: str,
    samples: int,
    epsilon: float,
    paraphrase: bool,
    scrub_pii: bool,
    audit: bool,
):
    """Generate privacy-safe fine-tuning data for LLMs.

    Example:
        genesis finetune-data training.jsonl -o safe_training.jsonl -n 50000 --epsilon 0.5
    """
    from genesis.llm_finetuning import (
        FineTuningDataGenerator,
        SafetyConfig,
    )

    # Load input data
    if input_file.endswith(".jsonl"):
        with open(input_file) as f:
            texts = [json.loads(line).get("text", "") for line in f]
    else:
        with open(input_file) as f:
            texts = f.read().splitlines()

    console.print(f"[bold]Generating safe fine-tuning data[/]")
    console.print(f"Input samples: {len(texts)}")
    console.print(f"Target samples: {samples}")
    console.print(f"Privacy budget (ε): {epsilon}")

    config = SafetyConfig(
        enable_dp=True,
        epsilon=epsilon,
        enable_paraphrase=paraphrase,
        remove_pii=scrub_pii,
    )

    generator = FineTuningDataGenerator(config=config)

    with console.status("Generating..."):
        safe_data = generator.generate_from_data(texts, n_samples=samples)

    console.print(f"[green]Generated {len(safe_data)} samples[/]")

    # Audit if requested
    if audit:
        with console.status("Running memorization audit..."):
            report = generator.audit_memorization(safe_data, texts)

        console.print(f"\n[bold]Memorization Audit[/]")
        console.print(f"Risk score: {report.risk_score:.4f}")
        console.print(f"Exact matches: {report.exact_matches}")
        console.print(f"Near duplicates: {report.near_duplicates}")
        console.print(f"Status: {'[green]PASSED[/]' if report.passed else '[red]FAILED[/]'}")

        if report.recommendations:
            console.print("\nRecommendations:")
            for rec in report.recommendations:
                console.print(f"  • {rec}")

    # Save output
    generator.export_for_training(safe_data, "jsonl", output)
    console.print(f"\n[green]Saved to: {output}[/]")


# Export all commands
__all__ = [
    "realtime",
    "cicd",
    "deploy",
    "dp_query_cmd",
    "leaderboard",
    "finetune_data",
]
