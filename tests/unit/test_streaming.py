"""Tests for streaming and incremental generation."""

import queue
import time

import numpy as np
import pandas as pd
import pytest

from genesis.streaming import (
    BatchIterator,
    DataStreamProcessor,
    StreamingConfig,
    StreamingGenerator,
    StreamingStats,
    generate_streaming,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric": np.random.randn(1000),
            "category": np.random.choice(["A", "B", "C"], 1000),
        }
    )


class TestStreamingConfig:
    """Tests for StreamingConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = StreamingConfig()

        assert config.batch_size == 100
        assert config.buffer_size == 10
        assert config.max_batches is None
        assert config.delay_seconds == 0.0

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = StreamingConfig(
            batch_size=500,
            buffer_size=20,
            max_batches=10,
        )

        assert config.batch_size == 500
        assert config.buffer_size == 20
        assert config.max_batches == 10


class TestStreamingStats:
    """Tests for StreamingStats."""

    def test_samples_per_second(self) -> None:
        """Test samples per second calculation."""
        stats = StreamingStats()
        stats.start_time = time.time() - 10  # 10 seconds ago
        stats.samples_generated = 1000

        sps = stats.samples_per_second
        assert 90 < sps < 110  # ~100 samples/sec

    def test_zero_samples(self) -> None:
        """Test with zero samples."""
        stats = StreamingStats()
        stats.start_time = time.time()
        stats.samples_generated = 0

        assert stats.samples_per_second == 0.0


class TestStreamingGenerator:
    """Tests for StreamingGenerator."""

    def test_fit(self, sample_data: pd.DataFrame) -> None:
        """Test fitting the generator."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        assert generator.is_fitted

    def test_generate(self, sample_data: pd.DataFrame) -> None:
        """Test generating synthetic data."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        synthetic = generator.generate(100)

        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(sample_data.columns)

    def test_generate_stream(self, sample_data: pd.DataFrame) -> None:
        """Test streaming generation."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        batches = list(generator.generate_stream(n_batches=3, batch_size=50))

        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 50

    def test_stats_updated(self, sample_data: pd.DataFrame) -> None:
        """Test that stats are updated during generation."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        initial_batches = generator.stats.batches_generated
        initial_samples = generator.stats.samples_generated

        generator.generate(100)

        assert generator.stats.batches_generated > initial_batches
        assert generator.stats.samples_generated > initial_samples

    def test_generate_not_fitted_raises(self) -> None:
        """Test that generate raises if not fitted."""
        from genesis.core.exceptions import NotFittedError

        generator = StreamingGenerator()

        with pytest.raises(NotFittedError):
            generator.generate(100)


class TestBatchIterator:
    """Tests for BatchIterator."""

    def test_iteration(self, sample_data: pd.DataFrame) -> None:
        """Test iterating through batches."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        iterator = BatchIterator(generator, total_samples=250, batch_size=100)

        batches = list(iterator)

        assert len(batches) == 3  # 100 + 100 + 50
        total_samples = sum(len(b) for b in batches)
        assert total_samples == 250

    def test_len(self, sample_data: pd.DataFrame) -> None:
        """Test length calculation."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        iterator = BatchIterator(generator, total_samples=250, batch_size=100)

        assert len(iterator) == 3


class TestDataStreamProcessor:
    """Tests for DataStreamProcessor."""

    def test_process(self, sample_data: pd.DataFrame) -> None:
        """Test processing incoming data."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        processor = DataStreamProcessor(
            generator,
            window_size=500,
            update_threshold=100,
        )

        # Process incoming data
        incoming = sample_data.head(150)
        synthetic = processor.process(incoming, generate_ratio=0.5)

        # Should generate 75 samples (150 * 0.5)
        assert len(synthetic) == 75


class TestGenerateStreaming:
    """Tests for generate_streaming convenience function."""

    def test_generates_correct_total(self, sample_data: pd.DataFrame) -> None:
        """Test that total samples is correct."""
        batches = list(
            generate_streaming(
                sample_data,
                n_samples=250,
                batch_size=100,
            )
        )

        total = sum(len(b) for b in batches)
        assert total == 250

    def test_respects_batch_size(self, sample_data: pd.DataFrame) -> None:
        """Test that batch sizes are respected."""
        batches = list(
            generate_streaming(
                sample_data,
                n_samples=500,
                batch_size=100,
            )
        )

        # All batches except last should be full size
        for batch in batches[:-1]:
            assert len(batch) == 100


class TestAsyncGeneration:
    """Tests for async generation capabilities."""

    def test_generate_to_queue(self, sample_data: pd.DataFrame) -> None:
        """Test generating to a queue."""
        generator = StreamingGenerator(method="gaussian_copula")
        generator.fit(sample_data)

        output_queue = queue.Queue()

        thread = generator.generate_to_queue(
            output_queue,
            n_batches=3,
            batch_size=50,
        )

        # Collect results
        batches = []
        while True:
            batch = output_queue.get(timeout=5)
            if batch is None:
                break
            batches.append(batch)

        thread.join(timeout=5)

        assert len(batches) == 3
        for batch in batches:
            assert len(batch) == 50
