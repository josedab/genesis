"""Tests for MLflow and W&B integrations."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# Mock the mlflow and wandb modules before importing
@pytest.fixture(autouse=True)
def mock_mlflow_wandb():
    """Mock mlflow and wandb for all tests."""
    mock_mlflow = MagicMock()
    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()
    mock_wandb.run.id = "test-run-id"

    with patch.dict(sys.modules, {"mlflow": mock_mlflow, "wandb": mock_wandb}):
        yield mock_mlflow, mock_wandb


class TestMLflowCallback:
    """Tests for MLflowCallback."""

    def test_callback_logs_at_frequency(self, mock_mlflow_wandb) -> None:
        """Test that callback logs at specified frequency."""
        mock_mlflow, _ = mock_mlflow_wandb

        from genesis.integrations.mlflow_integration import MLflowCallback

        callback = MLflowCallback(log_frequency=5, prefix="train")

        # Call 10 times
        for i in range(10):
            callback(epoch=0, step=i, metrics={"loss": 0.5})

        # Should have logged twice (at step 5 and 10)
        assert mock_mlflow.log_metric.call_count == 4  # 2 metrics * 2 times

    def test_callback_uses_prefix(self, mock_mlflow_wandb) -> None:
        """Test that callback uses metric prefix."""
        mock_mlflow, _ = mock_mlflow_wandb

        from genesis.integrations.mlflow_integration import MLflowCallback

        callback = MLflowCallback(log_frequency=1, prefix="custom")
        callback(epoch=1, step=0, metrics={"loss": 0.3})

        # Check that prefix is used
        calls = mock_mlflow.log_metric.call_args_list
        metric_names = [call[0][0] for call in calls]
        assert any("custom/" in name for name in metric_names)


class TestLogGeneratorToMLflow:
    """Tests for log_generator_to_mlflow function."""

    @pytest.fixture
    def mock_generator(self) -> MagicMock:
        """Create a mock generator."""
        generator = MagicMock()
        generator.get_parameters.return_value = {
            "is_fitted": True,
            "config": {"method": "ctgan", "epochs": 300},
            "privacy": {"epsilon": 1.0},
            "n_constraints": 0,
        }
        generator.__class__.__name__ = "MockGenerator"
        return generator

    def test_logs_parameters(self, mock_mlflow_wandb, mock_generator: MagicMock) -> None:
        """Test that generator parameters are logged."""
        mock_mlflow, _ = mock_mlflow_wandb
        mock_mlflow.active_run.return_value = None
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        from genesis.integrations.mlflow_integration import log_generator_to_mlflow

        run_id = log_generator_to_mlflow(
            mock_generator,
            log_model=False,
            log_data_sample=False,
        )

        assert run_id == "test-run-id"
        assert mock_mlflow.log_params.called

    def test_logs_synthetic_data_sample(self, mock_mlflow_wandb, mock_generator: MagicMock) -> None:
        """Test that synthetic data sample is logged."""
        mock_mlflow, _ = mock_mlflow_wandb
        mock_mlflow.active_run.return_value = None
        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        from genesis.integrations.mlflow_integration import log_generator_to_mlflow

        synthetic = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        log_generator_to_mlflow(
            mock_generator,
            synthetic_data=synthetic,
            log_model=False,
            log_data_sample=True,
            sample_size=2,
        )

        assert mock_mlflow.log_artifact.called


class TestMLflowExperimentTracker:
    """Tests for MLflowExperimentTracker context manager."""

    def test_context_manager_starts_and_ends_run(self, mock_mlflow_wandb) -> None:
        """Test that context manager properly manages MLflow run."""
        mock_mlflow, _ = mock_mlflow_wandb

        from genesis.integrations.mlflow_integration import MLflowExperimentTracker

        mock_run = MagicMock()
        mock_run.info.run_id = "test-id"
        mock_mlflow.start_run.return_value = mock_run

        with MLflowExperimentTracker("test-experiment", run_name="test-run") as tracker:
            assert tracker.run_id == "test-id"

        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.end_run.assert_called_once()


class TestWandbCallback:
    """Tests for WandbCallback."""

    def test_callback_logs_at_frequency(self, mock_mlflow_wandb) -> None:
        """Test that callback logs at specified frequency."""
        _, mock_wandb = mock_mlflow_wandb

        from genesis.integrations.wandb_integration import WandbCallback

        callback = WandbCallback(log_frequency=5, prefix="train")

        for i in range(10):
            callback(epoch=0, step=i, metrics={"loss": 0.5})

        # Should have logged twice
        assert mock_wandb.log.call_count == 2


class TestLogGeneratorToWandb:
    """Tests for log_generator_to_wandb function."""

    @pytest.fixture
    def mock_generator(self) -> MagicMock:
        """Create a mock generator."""
        generator = MagicMock()
        generator.get_parameters.return_value = {
            "is_fitted": True,
            "config": {"method": "ctgan", "epochs": 300},
            "privacy": {"epsilon": 1.0},
            "n_constraints": 0,
        }
        generator.__class__.__name__ = "MockGenerator"
        return generator

    def test_logs_config(self, mock_mlflow_wandb, mock_generator: MagicMock) -> None:
        """Test that generator config is logged."""
        _, mock_wandb = mock_mlflow_wandb

        from genesis.integrations.wandb_integration import log_generator_to_wandb

        log_generator_to_wandb(
            mock_generator,
            log_model=False,
            log_data_sample=False,
        )

        assert mock_wandb.config.update.called

    def test_logs_data_table(self, mock_mlflow_wandb, mock_generator: MagicMock) -> None:
        """Test that synthetic data is logged as table."""
        _, mock_wandb = mock_mlflow_wandb

        from genesis.integrations.wandb_integration import log_generator_to_wandb

        synthetic = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        log_generator_to_wandb(
            mock_generator,
            synthetic_data=synthetic,
            log_model=False,
            log_data_sample=True,
        )

        mock_wandb.Table.assert_called()


class TestWandbExperimentTracker:
    """Tests for WandbExperimentTracker context manager."""

    def test_context_manager_inits_and_finishes(self, mock_mlflow_wandb) -> None:
        """Test that context manager properly manages W&B run."""
        _, mock_wandb = mock_mlflow_wandb

        from genesis.integrations.wandb_integration import WandbExperimentTracker

        with WandbExperimentTracker(project="test-project", name="test-run") as tracker:
            assert tracker.run_id == "test-run-id"

        mock_wandb.init.assert_called_once()
        mock_wandb.finish.assert_called_once()


class TestCreateWandbSweepConfig:
    """Tests for sweep config creation."""

    def test_creates_valid_config(self, mock_mlflow_wandb) -> None:
        """Test that sweep config is valid."""
        from genesis.integrations.wandb_integration import create_wandb_sweep_config

        config = create_wandb_sweep_config(
            method_options=["ctgan", "tvae"],
            epoch_range=(100, 300),
            batch_size_options=[100, 200],
        )

        assert config["method"] == "bayes"
        assert "quality/overall_score" in config["metric"]["name"]
        assert "parameters" in config
        assert "method" in config["parameters"]
        assert "epochs" in config["parameters"]
