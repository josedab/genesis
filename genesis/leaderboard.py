"""Public Leaderboard for Synthetic Data Benchmarks.

Public leaderboard system for comparing synthetic data generators
across standardized datasets with submission, ranking, and display.

Features:
    - Standardized benchmark submission
    - Multi-metric ranking system
    - Public leaderboard website generation
    - Submission validation
    - Historical tracking
    - REST API for submissions

Example:
    Submit to leaderboard::

        from genesis.leaderboard import LeaderboardClient, Submission

        client = LeaderboardClient()
        
        submission = Submission(
            method_name="Genesis CTGAN",
            method_version="1.5.0",
            dataset="adult",
            metrics={
                "statistical_fidelity": 0.94,
                "privacy_score": 0.99,
                "ml_utility": 0.91,
            },
            organization="Acme Corp",
        )
        
        result = client.submit(submission)
        print(f"Rank: {result.rank}")

    Generate leaderboard HTML::

        from genesis.leaderboard import LeaderboardGenerator

        generator = LeaderboardGenerator()
        generator.generate_static_site("./leaderboard")

Classes:
    Submission: Benchmark submission.
    LeaderboardEntry: Entry in the leaderboard.
    Leaderboard: In-memory leaderboard.
    LeaderboardClient: Client for submitting results.
    LeaderboardGenerator: Static site generator.
    LeaderboardAPI: FastAPI endpoints for leaderboard.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class BenchmarkDataset(str, Enum):
    """Standard benchmark datasets."""

    ADULT = "adult"  # UCI Adult Income
    CENSUS = "census"  # US Census
    CREDIT = "credit"  # Credit Card Default
    DIABETES = "diabetes"  # Diabetes prediction
    COVERTYPE = "covertype"  # Forest cover type
    INTRUSION = "intrusion"  # Network intrusion
    NEWS = "news"  # News popularity
    LOAN = "loan"  # Loan default


class RankingMetric(str, Enum):
    """Metrics used for ranking."""

    STATISTICAL_FIDELITY = "statistical_fidelity"
    PRIVACY_SCORE = "privacy_score"
    ML_UTILITY = "ml_utility"
    TRAINING_TIME = "training_time"
    GENERATION_TIME = "generation_time"
    MEMORY_USAGE = "memory_usage"
    OVERALL_SCORE = "overall_score"


@dataclass
class MetricWeights:
    """Weights for computing overall score.

    Attributes:
        statistical_fidelity: Weight for fidelity (0-1)
        privacy_score: Weight for privacy (0-1)
        ml_utility: Weight for ML utility (0-1)
        efficiency: Weight for efficiency (0-1)
    """

    statistical_fidelity: float = 0.35
    privacy_score: float = 0.25
    ml_utility: float = 0.30
    efficiency: float = 0.10


@dataclass
class Submission:
    """Benchmark submission.

    Attributes:
        method_name: Name of the method
        method_version: Version string
        dataset: Dataset used
        metrics: Computed metrics
        organization: Submitting organization
        submitter_email: Contact email
        code_url: URL to code repository
        paper_url: URL to paper if applicable
        notes: Additional notes
    """

    method_name: str
    method_version: str
    dataset: Union[str, BenchmarkDataset]
    metrics: Dict[str, float]
    organization: str = ""
    submitter_email: str = ""
    code_url: str = ""
    paper_url: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def compute_overall_score(self, weights: Optional[MetricWeights] = None) -> float:
        """Compute overall score from metrics.

        Args:
            weights: Metric weights

        Returns:
            Overall score (0-1)
        """
        weights = weights or MetricWeights()

        score = 0.0
        score += self.metrics.get("statistical_fidelity", 0) * weights.statistical_fidelity
        score += self.metrics.get("privacy_score", 0) * weights.privacy_score
        score += self.metrics.get("ml_utility", 0) * weights.ml_utility

        # Efficiency is inverse (lower is better)
        training_time = self.metrics.get("training_time", 1000)
        efficiency = max(0, 1 - (training_time / 1000))  # Normalize to 0-1
        score += efficiency * weights.efficiency

        return round(score, 4)

    def validate(self) -> List[str]:
        """Validate submission.

        Returns:
            List of validation errors
        """
        errors = []

        if not self.method_name:
            errors.append("method_name is required")

        if not self.dataset:
            errors.append("dataset is required")

        required_metrics = ["statistical_fidelity", "privacy_score", "ml_utility"]
        for metric in required_metrics:
            if metric not in self.metrics:
                errors.append(f"Missing required metric: {metric}")
            elif not (0 <= self.metrics[metric] <= 1):
                errors.append(f"Metric {metric} must be between 0 and 1")

        return errors


@dataclass
class LeaderboardEntry:
    """Entry in the leaderboard.

    Attributes:
        submission_id: Unique submission ID
        rank: Current rank
        method_name: Name of method
        method_version: Version
        dataset: Dataset
        overall_score: Computed overall score
        metrics: All metrics
        organization: Organization
        submitted_at: Submission timestamp
        verified: Whether submission is verified
    """

    submission_id: str
    rank: int
    method_name: str
    method_version: str
    dataset: str
    overall_score: float
    metrics: Dict[str, float]
    organization: str = ""
    submitted_at: str = ""
    verified: bool = False
    badges: List[str] = field(default_factory=list)


class Leaderboard:
    """In-memory leaderboard with persistence.

    Maintains ranked list of submissions per dataset
    with automatic ranking updates.
    """

    def __init__(self, storage_path: str = "./leaderboard_data"):
        """Initialize leaderboard.

        Args:
            storage_path: Path for persistence
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Per-dataset entries
        self._entries: Dict[str, List[LeaderboardEntry]] = {}
        self._load()

    def submit(self, submission: Submission) -> LeaderboardEntry:
        """Submit a benchmark result.

        Args:
            submission: Benchmark submission

        Returns:
            LeaderboardEntry with rank

        Raises:
            ValueError: If submission is invalid
        """
        # Validate
        errors = submission.validate()
        if errors:
            raise ValueError(f"Invalid submission: {', '.join(errors)}")

        # Generate submission ID
        submission_id = hashlib.sha256(
            f"{submission.method_name}:{submission.method_version}:{submission.dataset}:{submission.timestamp}".encode()
        ).hexdigest()[:12]

        # Compute overall score
        overall_score = submission.compute_overall_score()

        # Create entry
        dataset = submission.dataset.value if isinstance(submission.dataset, BenchmarkDataset) else submission.dataset
        entry = LeaderboardEntry(
            submission_id=submission_id,
            rank=0,  # Will be computed
            method_name=submission.method_name,
            method_version=submission.method_version,
            dataset=dataset,
            overall_score=overall_score,
            metrics=submission.metrics,
            organization=submission.organization,
            submitted_at=submission.timestamp,
        )

        # Add to leaderboard
        if dataset not in self._entries:
            self._entries[dataset] = []

        self._entries[dataset].append(entry)

        # Recompute ranks
        self._update_ranks(dataset)

        # Persist
        self._save()

        # Return entry with computed rank
        return self.get_entry(submission_id)

    def _update_ranks(self, dataset: str) -> None:
        """Update ranks for a dataset."""
        if dataset not in self._entries:
            return

        # Sort by overall score descending
        self._entries[dataset].sort(key=lambda e: e.overall_score, reverse=True)

        # Assign ranks
        for i, entry in enumerate(self._entries[dataset]):
            entry.rank = i + 1

            # Assign badges
            entry.badges = []
            if i == 0:
                entry.badges.append("🥇 First Place")
            elif i == 1:
                entry.badges.append("🥈 Second Place")
            elif i == 2:
                entry.badges.append("🥉 Third Place")

            if entry.metrics.get("privacy_score", 0) >= 0.99:
                entry.badges.append("🔒 Privacy Champion")

            if entry.metrics.get("statistical_fidelity", 0) >= 0.95:
                entry.badges.append("📊 High Fidelity")

    def get_leaderboard(
        self,
        dataset: Optional[str] = None,
        top_k: int = 100,
    ) -> Dict[str, List[LeaderboardEntry]]:
        """Get leaderboard entries.

        Args:
            dataset: Filter by dataset (None = all)
            top_k: Number of entries per dataset

        Returns:
            Dict of dataset -> entries
        """
        if dataset:
            entries = self._entries.get(dataset, [])
            return {dataset: entries[:top_k]}

        return {d: entries[:top_k] for d, entries in self._entries.items()}

    def get_entry(self, submission_id: str) -> Optional[LeaderboardEntry]:
        """Get a specific entry.

        Args:
            submission_id: Submission ID

        Returns:
            LeaderboardEntry or None
        """
        for entries in self._entries.values():
            for entry in entries:
                if entry.submission_id == submission_id:
                    return entry
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get leaderboard statistics.

        Returns:
            Statistics dict
        """
        total_submissions = sum(len(entries) for entries in self._entries.values())
        organizations = set()
        methods = set()

        for entries in self._entries.values():
            for entry in entries:
                if entry.organization:
                    organizations.add(entry.organization)
                methods.add(entry.method_name)

        return {
            "total_submissions": total_submissions,
            "datasets": len(self._entries),
            "unique_organizations": len(organizations),
            "unique_methods": len(methods),
            "last_updated": datetime.utcnow().isoformat(),
        }

    def _save(self) -> None:
        """Save leaderboard to disk."""
        data = {
            dataset: [
                {
                    "submission_id": e.submission_id,
                    "rank": e.rank,
                    "method_name": e.method_name,
                    "method_version": e.method_version,
                    "dataset": e.dataset,
                    "overall_score": e.overall_score,
                    "metrics": e.metrics,
                    "organization": e.organization,
                    "submitted_at": e.submitted_at,
                    "verified": e.verified,
                    "badges": e.badges,
                }
                for e in entries
            ]
            for dataset, entries in self._entries.items()
        }

        with open(self._storage_path / "leaderboard.json", "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load leaderboard from disk."""
        leaderboard_file = self._storage_path / "leaderboard.json"
        if not leaderboard_file.exists():
            return

        with open(leaderboard_file) as f:
            data = json.load(f)

        for dataset, entries in data.items():
            self._entries[dataset] = [
                LeaderboardEntry(**entry)
                for entry in entries
            ]


class LeaderboardGenerator:
    """Generate static leaderboard website."""

    def __init__(self, leaderboard: Optional[Leaderboard] = None):
        """Initialize generator.

        Args:
            leaderboard: Leaderboard instance
        """
        self._leaderboard = leaderboard or Leaderboard()

    def generate_static_site(self, output_dir: str) -> str:
        """Generate static HTML leaderboard site.

        Args:
            output_dir: Output directory

        Returns:
            Path to index.html
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get data
        leaderboard_data = self._leaderboard.get_leaderboard()
        stats = self._leaderboard.get_stats()

        # Generate HTML
        html = self._generate_html(leaderboard_data, stats)

        # Write files
        index_path = output_path / "index.html"
        index_path.write_text(html)

        # Write CSS
        (output_path / "style.css").write_text(self._generate_css())

        # Write JSON data for JavaScript
        (output_path / "data.json").write_text(json.dumps(leaderboard_data, indent=2, default=self._json_serializer))

        logger.info(f"Generated leaderboard at {output_path}")
        return str(index_path)

    def _json_serializer(self, obj: Any) -> Any:
        """JSON serializer for dataclasses."""
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def _generate_html(
        self,
        leaderboard_data: Dict[str, List[LeaderboardEntry]],
        stats: Dict[str, Any],
    ) -> str:
        """Generate HTML content."""
        datasets_html = []
        for dataset, entries in leaderboard_data.items():
            rows = []
            for entry in entries[:20]:  # Top 20
                badges_html = " ".join(f'<span class="badge">{b}</span>' for b in entry.badges)
                rows.append(f"""
                    <tr>
                        <td class="rank">#{entry.rank}</td>
                        <td>
                            <strong>{entry.method_name}</strong>
                            <span class="version">v{entry.method_version}</span>
                            {badges_html}
                        </td>
                        <td class="org">{entry.organization or '-'}</td>
                        <td class="score">{entry.overall_score:.4f}</td>
                        <td class="metric">{entry.metrics.get('statistical_fidelity', 0):.3f}</td>
                        <td class="metric">{entry.metrics.get('privacy_score', 0):.3f}</td>
                        <td class="metric">{entry.metrics.get('ml_utility', 0):.3f}</td>
                    </tr>
                """)

            datasets_html.append(f"""
                <section class="dataset" id="{dataset}">
                    <h2>{dataset.upper()} Dataset</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Method</th>
                                <th>Organization</th>
                                <th>Overall</th>
                                <th>Fidelity</th>
                                <th>Privacy</th>
                                <th>ML Utility</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join(rows)}
                        </tbody>
                    </table>
                </section>
            """)

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Genesis Synthetic Data Benchmark Leaderboard</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <header>
        <h1>🏆 Genesis Synthetic Data Benchmark</h1>
        <p>Standardized comparison of synthetic data generators</p>
        <div class="stats">
            <span><strong>{stats['total_submissions']}</strong> submissions</span>
            <span><strong>{stats['datasets']}</strong> datasets</span>
            <span><strong>{stats['unique_methods']}</strong> methods</span>
            <span><strong>{stats['unique_organizations']}</strong> organizations</span>
        </div>
    </header>

    <nav>
        <h3>Datasets</h3>
        <ul>
            {''.join(f'<li><a href="#{d}">{d.upper()}</a></li>' for d in leaderboard_data.keys())}
        </ul>
    </nav>

    <main>
        {''.join(datasets_html)}
    </main>

    <footer>
        <h3>How to Submit</h3>
        <pre><code>from genesis.leaderboard import LeaderboardClient, Submission

client = LeaderboardClient()
submission = Submission(
    method_name="Your Method",
    method_version="1.0.0",
    dataset="adult",
    metrics={{
        "statistical_fidelity": 0.94,
        "privacy_score": 0.99,
        "ml_utility": 0.91,
    }},
)
client.submit(submission)</code></pre>
        <p>Last updated: {stats['last_updated']}</p>
    </footer>
</body>
</html>
"""

    def _generate_css(self) -> str:
        """Generate CSS styles."""
        return """
:root {
    --primary: #2563eb;
    --secondary: #10b981;
    --bg: #f8fafc;
    --card: #ffffff;
    --text: #1e293b;
    --muted: #64748b;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}

header {
    background: linear-gradient(135deg, var(--primary), #1e40af);
    color: white;
    padding: 3rem 2rem;
    text-align: center;
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.stats {
    display: flex;
    justify-content: center;
    gap: 2rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.stats span {
    background: rgba(255,255,255,0.2);
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
}

nav {
    background: var(--card);
    padding: 1rem 2rem;
    border-bottom: 1px solid #e2e8f0;
    position: sticky;
    top: 0;
    z-index: 100;
}

nav ul {
    display: flex;
    list-style: none;
    gap: 1rem;
    justify-content: center;
}

nav a {
    color: var(--primary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
}

nav a:hover {
    background: var(--bg);
}

main {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.dataset {
    background: var(--card);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.dataset h2 {
    margin-bottom: 1rem;
    color: var(--primary);
}

table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}

th {
    background: var(--bg);
    font-weight: 600;
    color: var(--muted);
    font-size: 0.875rem;
    text-transform: uppercase;
}

tr:hover {
    background: #f1f5f9;
}

.rank {
    font-weight: 700;
    color: var(--primary);
}

.version {
    font-size: 0.75rem;
    color: var(--muted);
    margin-left: 0.5rem;
}

.badge {
    display: inline-block;
    font-size: 0.625rem;
    background: #fef3c7;
    padding: 0.125rem 0.375rem;
    border-radius: 9999px;
    margin-left: 0.25rem;
}

.score {
    font-weight: 700;
    color: var(--secondary);
}

.metric {
    font-family: monospace;
}

.org {
    color: var(--muted);
    font-size: 0.875rem;
}

footer {
    background: var(--card);
    padding: 2rem;
    margin-top: 2rem;
    text-align: center;
}

footer h3 {
    margin-bottom: 1rem;
}

footer pre {
    background: #1e293b;
    color: #e2e8f0;
    padding: 1rem;
    border-radius: 0.5rem;
    overflow-x: auto;
    text-align: left;
    max-width: 800px;
    margin: 0 auto 1rem;
}

footer p {
    color: var(--muted);
    font-size: 0.875rem;
}

@media (max-width: 768px) {
    header h1 { font-size: 1.5rem; }
    .stats { flex-direction: column; gap: 0.5rem; }
    th, td { padding: 0.5rem; font-size: 0.875rem; }
}
"""


class LeaderboardClient:
    """Client for submitting to the leaderboard."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        local_leaderboard: Optional[Leaderboard] = None,
    ):
        """Initialize client.

        Args:
            api_url: URL of leaderboard API (None = local)
            local_leaderboard: Local leaderboard instance
        """
        self._api_url = api_url
        self._local = local_leaderboard or Leaderboard()

    def submit(self, submission: Submission) -> LeaderboardEntry:
        """Submit benchmark results.

        Args:
            submission: Benchmark submission

        Returns:
            LeaderboardEntry with rank
        """
        if self._api_url:
            return self._submit_remote(submission)
        return self._local.submit(submission)

    def _submit_remote(self, submission: Submission) -> LeaderboardEntry:
        """Submit to remote API."""
        try:
            import httpx

            response = httpx.post(
                f"{self._api_url}/v1/leaderboard/submit",
                json={
                    "method_name": submission.method_name,
                    "method_version": submission.method_version,
                    "dataset": submission.dataset.value if isinstance(submission.dataset, BenchmarkDataset) else submission.dataset,
                    "metrics": submission.metrics,
                    "organization": submission.organization,
                    "submitter_email": submission.submitter_email,
                    "code_url": submission.code_url,
                    "paper_url": submission.paper_url,
                    "notes": submission.notes,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            return LeaderboardEntry(**data)

        except ImportError:
            raise ImportError("httpx required: pip install httpx")

    def get_leaderboard(
        self,
        dataset: Optional[str] = None,
    ) -> Dict[str, List[LeaderboardEntry]]:
        """Get current leaderboard.

        Args:
            dataset: Filter by dataset

        Returns:
            Leaderboard data
        """
        if self._api_url:
            try:
                import httpx

                params = {"dataset": dataset} if dataset else {}
                response = httpx.get(
                    f"{self._api_url}/v1/leaderboard",
                    params=params,
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
            except ImportError:
                raise ImportError("httpx required: pip install httpx")

        return self._local.get_leaderboard(dataset)


def create_leaderboard_api() -> Any:
    """Create FastAPI app for leaderboard.

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("FastAPI required: pip install fastapi")

    app = FastAPI(
        title="Genesis Benchmark Leaderboard",
        description="Public leaderboard for synthetic data benchmarks",
    )

    leaderboard = Leaderboard()

    class SubmissionRequest(BaseModel):
        method_name: str
        method_version: str
        dataset: str
        metrics: Dict[str, float]
        organization: str = ""
        submitter_email: str = ""
        code_url: str = ""
        paper_url: str = ""
        notes: str = ""

    @app.post("/v1/leaderboard/submit")
    async def submit(request: SubmissionRequest):
        """Submit benchmark results."""
        try:
            submission = Submission(
                method_name=request.method_name,
                method_version=request.method_version,
                dataset=request.dataset,
                metrics=request.metrics,
                organization=request.organization,
                submitter_email=request.submitter_email,
                code_url=request.code_url,
                paper_url=request.paper_url,
                notes=request.notes,
            )
            entry = leaderboard.submit(submission)
            return {
                "submission_id": entry.submission_id,
                "rank": entry.rank,
                "overall_score": entry.overall_score,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/leaderboard")
    async def get_leaderboard(dataset: Optional[str] = None, top_k: int = 100):
        """Get leaderboard."""
        return leaderboard.get_leaderboard(dataset, top_k)

    @app.get("/v1/leaderboard/stats")
    async def get_stats():
        """Get leaderboard statistics."""
        return leaderboard.get_stats()

    return app


# Convenience function for running benchmark and submitting
def run_and_submit(
    generator: Any,
    dataset: Union[str, BenchmarkDataset],
    method_name: str,
    method_version: str = "1.0.0",
    organization: str = "",
    n_samples: int = 10000,
    leaderboard_url: Optional[str] = None,
) -> LeaderboardEntry:
    """Run benchmark and submit to leaderboard.

    Args:
        generator: Fitted generator
        dataset: Benchmark dataset
        method_name: Method name
        method_version: Method version
        organization: Organization name
        n_samples: Number of samples
        leaderboard_url: Leaderboard API URL

    Returns:
        LeaderboardEntry with rank
    """
    from genesis.benchmarking import BenchmarkSuite

    # Run benchmark
    suite = BenchmarkSuite()
    results = suite.run_single(
        dataset=dataset.value if isinstance(dataset, BenchmarkDataset) else dataset,
        generator=generator,
        n_samples=n_samples,
    )

    # Create submission
    submission = Submission(
        method_name=method_name,
        method_version=method_version,
        dataset=dataset,
        metrics={
            "statistical_fidelity": results.statistical_fidelity,
            "privacy_score": results.privacy_score,
            "ml_utility": results.ml_utility,
            "training_time": results.training_time,
            "generation_time": results.generation_time,
        },
        organization=organization,
    )

    # Submit
    client = LeaderboardClient(api_url=leaderboard_url)
    return client.submit(submission)
