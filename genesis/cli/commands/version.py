"""Version control command for Genesis CLI."""

import click
from rich.console import Console

console = Console()


@click.command()
@click.argument("action", type=click.Choice(["init", "commit", "log", "tag", "checkout", "diff"]))
@click.option("--repo", "-r", "repo_path", default="./data_repo", help="Repository path")
@click.option("--data", "-d", "data_path", default=None, help="Data file path")
@click.option("--message", "-m", default=None, help="Commit/tag message")
@click.option("--output", "-o", "output_path", default=None, help="Output path for checkout")
@click.option("--ref1", default=None, help="First reference for diff")
@click.option("--ref2", default=None, help="Second reference for diff")
def version(
    action: str,
    repo_path: str,
    data_path: str,
    message: str,
    output_path: str,
    ref1: str,
    ref2: str,
):
    """Git-like versioning for synthetic datasets.

    Example:
        genesis version init ./data_repo
        genesis version commit -r ./data_repo -d synthetic.csv -m "Initial commit"
        genesis version log -r ./data_repo
        genesis version tag -r ./data_repo v1.0 -m "Release"
    """
    import pandas as pd

    from genesis.versioning import DatasetRepository

    console.print("[bold blue]Genesis Versioning[/]")

    if action == "init":
        repo = DatasetRepository.init(repo_path)
        console.print(f"[bold green]✓[/] Initialized repository at {repo_path}")

    elif action == "commit":
        if not data_path:
            console.print("[red]Error: --data required for commit[/]")
            return
        repo = DatasetRepository(repo_path)
        df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)
        commit = repo.commit(df, message=message or "No message")
        console.print(f"[bold green]✓[/] Committed: {commit.hash[:8]}")

    elif action == "log":
        repo = DatasetRepository(repo_path)
        for commit in repo.log()[:10]:
            console.print(f"  {commit.hash[:8]} - {commit.message} ({commit.timestamp})")

    elif action == "tag":
        if not message:
            console.print("[red]Error: --message required for tag[/]")
            return
        repo = DatasetRepository(repo_path)
        # The message is used as tag name, actual message is optional
        # Assuming tag name is passed via a different mechanism or reuse message
        repo.tag(message)
        console.print(f"[bold green]✓[/] Created tag: {message}")

    elif action == "checkout":
        if not ref1:
            console.print("[red]Error: Provide ref to checkout (e.g., --ref1 v1.0)[/]")
            return
        repo = DatasetRepository(repo_path)
        repo.checkout(ref1)
        if output_path:
            df = repo.get_current_data()
            if output_path.endswith(".csv"):
                df.to_csv(output_path, index=False)
            else:
                df.to_parquet(output_path, index=False)
            console.print(f"[bold green]✓[/] Exported to {output_path}")
        else:
            console.print(f"[bold green]✓[/] Checked out: {ref1}")

    elif action == "diff":
        if not ref1 or not ref2:
            console.print("[red]Error: --ref1 and --ref2 required for diff[/]")
            return
        repo = DatasetRepository(repo_path)
        diff = repo.diff(ref1, ref2)
        console.print(f"[bold]Diff {ref1}..{ref2}:[/]")
        console.print(f"  Rows added: {diff.rows_added}")
        console.print(f"  Rows removed: {diff.rows_removed}")
        console.print(f"  Columns changed: {diff.columns_changed}")
