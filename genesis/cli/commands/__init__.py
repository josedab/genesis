"""CLI command modules for Genesis."""

from genesis.cli.commands.analyze import analyze, privacy_audit
from genesis.cli.commands.automl import automl, discover
from genesis.cli.commands.data import augment, domain, pipeline
from genesis.cli.commands.evaluate import drift, evaluate, report
from genesis.cli.commands.generate import generate
from genesis.cli.commands.interactive import chat, dashboard
from genesis.cli.commands.version import version

__all__ = [
    "analyze",
    "augment",
    "automl",
    "chat",
    "dashboard",
    "discover",
    "domain",
    "drift",
    "evaluate",
    "generate",
    "pipeline",
    "privacy_audit",
    "report",
    "version",
]
