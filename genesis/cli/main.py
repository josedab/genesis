"""CLI entry point for Genesis."""

import click
from rich.console import Console

from genesis.cli.commands.analyze import analyze, privacy_audit
from genesis.cli.commands.automl import automl, discover
from genesis.cli.commands.data import augment, domain, pipeline
from genesis.cli.commands.evaluate import drift, evaluate, report
from genesis.cli.commands.generate import generate
from genesis.cli.commands.interactive import chat, dashboard
from genesis.cli.commands.nextgen import (
    cicd,
    deploy,
    dp_query_cmd,
    finetune_data,
    leaderboard,
    realtime,
)
from genesis.cli.commands.streaming import stream
from genesis.cli.commands.version import version
from genesis.version import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def main():
    """Genesis: Synthetic Data Generation Platform.

    Generate realistic, privacy-safe synthetic data for ML training,
    testing, and development.
    """
    pass


# Register all commands
main.add_command(generate)
main.add_command(evaluate)
main.add_command(analyze)
main.add_command(report)
main.add_command(chat)
main.add_command(dashboard)
main.add_command(discover)
main.add_command(automl)
main.add_command(augment)
main.add_command(privacy_audit)
main.add_command(drift)
main.add_command(version)
main.add_command(domain)
main.add_command(pipeline)
main.add_command(stream)

# Next-gen commands (v2.1.0)
main.add_command(realtime)
main.add_command(cicd)
main.add_command(deploy)
main.add_command(dp_query_cmd, name="dp-query")
main.add_command(leaderboard)
main.add_command(finetune_data, name="finetune-data")


if __name__ == "__main__":
    main()
