"""Streaming and incremental synthetic data generation.

This package provides capabilities for:
- Streaming generation (generate data continuously)
- Incremental model updates (update models without full retraining)
- Online learning for synthetic data
- Kafka and WebSocket integrations

Example:
    >>> from genesis.streaming import StreamingGenerator
    >>>
    >>> # Create streaming generator
    >>> stream = StreamingGenerator(method='gaussian_copula')
    >>> stream.fit(initial_data)
    >>>
    >>> # Generate data in batches
    >>> for batch in stream.generate_stream(n_batches=100, batch_size=100):
    ...     process(batch)
    >>>
    >>> # Update model incrementally
    >>> stream.partial_fit(new_data)
"""

from genesis.streaming.config import StreamingConfig, StreamingStats
from genesis.streaming.generator import (
    BatchIterator,
    DataStreamProcessor,
    StreamingGenerator,
    generate_streaming,
)
from genesis.streaming.kafka import KafkaStreamingGenerator
from genesis.streaming.websocket import WebSocketStreamingGenerator

__all__ = [
    # Configuration
    "StreamingConfig",
    "StreamingStats",
    # Core streaming
    "StreamingGenerator",
    "DataStreamProcessor",
    "BatchIterator",
    "generate_streaming",
    # Integrations
    "KafkaStreamingGenerator",
    "WebSocketStreamingGenerator",
]
