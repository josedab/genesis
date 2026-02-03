"""Streaming commands for Genesis CLI."""

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def stream():
    """Production streaming commands.

    Stream synthetic data to Kafka with production-grade features:
    - Exactly-once delivery semantics
    - Checkpoint management for fault tolerance
    - Rate limiting and backpressure handling
    - Dead letter queue for failed records
    """
    pass


@stream.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to training data")
@click.option("--topic", "-t", required=True, help="Kafka topic to produce to")
@click.option(
    "--bootstrap-servers", "-b", default="localhost:9092", help="Kafka bootstrap servers"
)
@click.option("--method", "-m", default="gaussian_copula", help="Generation method")
@click.option("--rate", "-r", type=float, default=None, help="Records per second (unlimited if not set)")
@click.option("--total", "-n", type=int, default=None, help="Total records to produce (unlimited if not set)")
@click.option("--duration", "-d", type=int, default=None, help="Duration in seconds (unlimited if not set)")
@click.option("--batch-size", default=100, type=int, help="Batch size for generation")
@click.option("--checkpoint-dir", default="./checkpoints", help="Directory for checkpoints")
@click.option("--checkpoint-interval", default=1000, type=int, help="Records between checkpoints")
@click.option(
    "--delivery",
    type=click.Choice(["at_most_once", "at_least_once", "exactly_once"]),
    default="exactly_once",
    help="Delivery semantics",
)
@click.option("--enable-dlq/--no-dlq", default=True, help="Enable dead letter queue")
@click.option("--discrete", "-c", multiple=True, help="Discrete columns")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def produce(
    input_path: str,
    topic: str,
    bootstrap_servers: str,
    method: str,
    rate: float,
    total: int,
    duration: int,
    batch_size: int,
    checkpoint_dir: str,
    checkpoint_interval: int,
    delivery: str,
    enable_dlq: bool,
    discrete: tuple,
    verbose: bool,
):
    """Produce synthetic data to Kafka with production features.

    Example:
        genesis stream produce -i data.csv -t synth-data -r 1000 -n 100000
    """
    import pandas as pd

    from genesis.streaming import StreamingGenerator
    from genesis.streaming.production import (
        DeliverySemantics,
        ProducerConfig,
        ProductionKafkaProducer,
    )

    if verbose:
        console.print("[bold blue]Genesis Production Streaming[/]")
        console.print(f"Topic: {topic}")
        console.print(f"Bootstrap servers: {bootstrap_servers}")

    # Load training data
    if verbose:
        console.print(f"Loading training data from {input_path}...")

    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)

    # Create and fit generator
    if verbose:
        console.print(f"Fitting generator with {len(df)} samples...")

    generator = StreamingGenerator(method=method)
    discrete_cols = list(discrete) if discrete else None
    generator.fit(df, discrete_columns=discrete_cols)

    # Configure producer
    delivery_map = {
        "at_most_once": DeliverySemantics.AT_MOST_ONCE,
        "at_least_once": DeliverySemantics.AT_LEAST_ONCE,
        "exactly_once": DeliverySemantics.EXACTLY_ONCE,
    }

    config = ProducerConfig(
        bootstrap_servers=bootstrap_servers,
        topic=topic,
        delivery_semantics=delivery_map[delivery],
        rate_limit=rate,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        dlq_enabled=enable_dlq,
    )

    producer = ProductionKafkaProducer(config, generator)

    if verbose:
        console.print("[bold green]Starting producer...[/]")
        if rate:
            console.print(f"Rate limit: {rate} records/sec")
        if total:
            console.print(f"Target records: {total}")
        if duration:
            console.print(f"Duration: {duration} seconds")

    try:
        producer.start()

        import time
        start_time = time.time()
        produced = 0
        target = total or float("inf")

        with console.status("[bold green]Producing...") as status:
            while produced < target:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break

                # Produce batch
                batch_target = min(batch_size, target - produced)
                batch_produced = producer.produce(batch_target)
                produced += batch_produced

                # Update status
                metrics = producer.get_metrics()
                status.update(
                    f"[bold green]Producing... "
                    f"{produced:,} records | "
                    f"{metrics['throughput_per_second']:.1f} rec/s | "
                    f"{metrics['average_latency_ms']:.1f}ms latency"
                )

        producer.stop()

        # Print final metrics
        if verbose:
            metrics = producer.get_metrics()
            console.print("\n[bold green]✓ Production complete[/]")

            table = Table(title="Final Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Records Produced", f"{metrics['records_produced']:,}")
            table.add_row("Records Failed", f"{metrics['records_failed']:,}")
            table.add_row("Records in DLQ", f"{metrics['records_in_dlq']:,}")
            table.add_row("Bytes Produced", f"{metrics['bytes_produced']:,}")
            table.add_row("Checkpoints Created", f"{metrics['checkpoints_created']}")
            table.add_row("Avg Latency", f"{metrics['average_latency_ms']:.2f} ms")
            table.add_row("Throughput", f"{metrics['throughput_per_second']:.2f} rec/s")
            table.add_row("Backpressure Events", f"{metrics['backpressure_events']}")

            console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Stopping gracefully...[/]")
        producer.stop()
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/]")
        raise click.Abort()


@stream.command()
@click.option("--checkpoint-dir", "-c", default="./checkpoints", help="Checkpoint directory")
def checkpoints(checkpoint_dir: str):
    """List and manage checkpoints.

    Example:
        genesis stream checkpoints -c ./checkpoints
    """
    from pathlib import Path

    from genesis.streaming.production import CheckpointManager

    manager = CheckpointManager(checkpoint_dir)
    latest = manager.get_latest_checkpoint()

    if not latest:
        console.print("[yellow]No checkpoints found[/]")
        return

    console.print(f"[bold blue]Latest Checkpoint[/]: {latest.checkpoint_id}")
    console.print(f"  Sequence: {latest.sequence_number}")
    console.print(f"  Committed Offset: {latest.committed_offset}")
    console.print(f"  Pending Records: {latest.pending_records}")

    import datetime
    ts = datetime.datetime.fromtimestamp(latest.timestamp)
    console.print(f"  Timestamp: {ts.isoformat()}")

    # List all checkpoints
    checkpoint_path = Path(checkpoint_dir)
    all_checkpoints = sorted(checkpoint_path.glob("ckpt_*.json"))

    console.print(f"\n[bold]Total checkpoints:[/] {len(all_checkpoints)}")

    table = Table(title="All Checkpoints")
    table.add_column("Checkpoint ID")
    table.add_column("Sequence")
    table.add_column("Offset")
    table.add_column("Timestamp")

    import json
    for cp_path in all_checkpoints[-10:]:  # Show last 10
        with open(cp_path) as f:
            cp = json.load(f)
        ts = datetime.datetime.fromtimestamp(cp["timestamp"])
        table.add_row(
            cp["checkpoint_id"],
            str(cp["sequence_number"]),
            str(cp["committed_offset"]),
            ts.strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)


@stream.command()
@click.option("--checkpoint-dir", "-c", default="./checkpoints", help="Checkpoint directory")
@click.option("--keep", "-k", default=5, type=int, help="Number of checkpoints to keep")
def cleanup(checkpoint_dir: str, keep: int):
    """Clean up old checkpoints.

    Example:
        genesis stream cleanup -c ./checkpoints -k 5
    """
    from genesis.streaming.production import CheckpointManager

    manager = CheckpointManager(checkpoint_dir)
    removed = manager.cleanup_old_checkpoints(keep=keep)

    console.print(f"[green]Removed {removed} old checkpoints[/]")
    console.print(f"Kept {keep} most recent checkpoints")


@stream.command()
@click.option("--input", "-i", "input_path", required=True, help="Path to training data")
@click.option("--port", "-p", default=8765, type=int, help="WebSocket port")
@click.option("--host", "-h", "host", default="localhost", help="Host to bind to")
@click.option("--method", "-m", default="gaussian_copula", help="Generation method")
@click.option("--rate", "-r", type=float, default=10.0, help="Records per second")
@click.option("--discrete", "-c", multiple=True, help="Discrete columns")
def websocket(
    input_path: str,
    port: int,
    host: str,
    method: str,
    rate: float,
    discrete: tuple,
):
    """Start a WebSocket server for streaming synthetic data.

    Example:
        genesis stream websocket -i data.csv -p 8765 -r 100
    """
    import pandas as pd

    from genesis.streaming import StreamingGenerator, WebSocketStreamingGenerator

    console.print("[bold blue]Genesis WebSocket Streaming Server[/]")
    console.print(f"Loading training data from {input_path}...")

    df = pd.read_csv(input_path) if input_path.endswith(".csv") else pd.read_parquet(input_path)

    console.print(f"Fitting generator with {len(df)} samples...")

    # Create base generator
    base_generator = StreamingGenerator(method=method)
    discrete_cols = list(discrete) if discrete else None
    base_generator.fit(df, discrete_columns=discrete_cols)

    console.print(f"\n[bold green]Starting WebSocket server at ws://{host}:{port}[/]")
    console.print(f"Rate: {rate} records/second")
    console.print("\nPress Ctrl+C to stop\n")

    try:
        ws_generator = WebSocketStreamingGenerator(
            host=host,
            port=port,
            generator=base_generator,
        )
        ws_generator.start(rate=rate)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped[/]")
