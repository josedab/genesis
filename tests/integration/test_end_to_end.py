"""End-to-end integration tests."""



class TestEndToEndTabular:
    """End-to-end tests for tabular data generation."""

    def test_gaussian_copula_full_pipeline(self, sample_mixed_df):
        """Test complete pipeline with Gaussian Copula."""
        from genesis import Constraint, QualityEvaluator, SyntheticGenerator

        # Generate
        generator = SyntheticGenerator(method="gaussian_copula")
        generator.fit(
            sample_mixed_df,
            discrete_columns=["gender", "city", "active"],
            constraints=[Constraint.positive("age")],
        )

        synthetic = generator.generate(n_samples=100)

        # Verify output
        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(sample_mixed_df.columns)
        assert (synthetic["age"] > 0).all()  # Constraint applied

        # Evaluate
        evaluator = QualityEvaluator(sample_mixed_df, synthetic)
        report = evaluator.evaluate()

        assert report.overall_score > 0
        assert report.fidelity_score > 0

    def test_fit_generate_convenience(self, sample_numeric_df):
        """Test fit_generate convenience method."""
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator(verbose=False)
        synthetic = generator.fit_generate(sample_numeric_df, n_samples=50)

        assert len(synthetic) == 50
        assert set(synthetic.columns) == set(sample_numeric_df.columns)


class TestEndToEndTimeSeries:
    """End-to-end tests for time series generation."""

    def test_statistical_timeseries(self, sample_timeseries_df):
        """Test statistical time series generation."""
        from genesis.generators.timeseries import StatisticalTimeSeriesGenerator

        generator = StatisticalTimeSeriesGenerator(verbose=False)
        generator.fit(sample_timeseries_df)

        synthetic = generator.generate(n_samples=100)

        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(sample_timeseries_df.columns)


class TestEndToEndMultiTable:
    """End-to-end tests for multi-table generation."""

    def test_multitable_generation(self, sample_multitable):
        """Test multi-table generation with foreign keys."""
        from genesis.multitable import MultiTableGenerator, RelationalSchema

        users = sample_multitable["users"]
        orders = sample_multitable["orders"]

        # Create schema
        schema = RelationalSchema.from_dataframes(
            sample_multitable,
            foreign_keys=[
                {
                    "child_table": "orders",
                    "child_column": "user_id",
                    "parent_table": "users",
                    "parent_column": "user_id",
                }
            ],
        )

        # Generate
        generator = MultiTableGenerator(verbose=False)
        generator.fit_tables(sample_multitable, schema)

        synthetic = generator.generate_tables(n_samples={"users": 10, "orders": 30})

        # Verify
        assert len(synthetic["users"]) == 10
        assert len(synthetic["orders"]) == 30

        # Check referential integrity
        syn_user_ids = set(synthetic["users"]["user_id"])
        order_user_ids = set(synthetic["orders"]["user_id"])
        assert order_user_ids.issubset(syn_user_ids)


class TestQualityReport:
    """Tests for quality report generation."""

    def test_full_report_generation(self, sample_mixed_df):
        """Test complete quality report generation."""
        from genesis import QualityEvaluator

        # Create synthetic with minor modifications
        synthetic = sample_mixed_df.copy()
        synthetic["income"] = synthetic["income"] * 1.05

        evaluator = QualityEvaluator(sample_mixed_df, synthetic)
        report = evaluator.evaluate(target_column="active")

        # Test all report methods
        summary = report.summary()
        assert len(summary) > 0

        json_report = report.to_json()
        assert "overall_score" in json_report

        html_report = report.to_html()
        assert "<html>" in html_report
