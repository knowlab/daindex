import warnings
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from daindex.main import DAIndex, DeteriorationFeature, Group


@pytest.fixture
def sample_cohort() -> pd.DataFrame:
    """Create a sample cohort DataFrame for testing."""
    np.random.seed(42)
    n_samples = 200

    return pd.DataFrame(
        {
            "group": ["A"] * 100 + ["B"] * 100,
            "deterioration_score": np.concatenate(
                [
                    np.random.normal(5, 2, 100),  # Group A
                    np.random.normal(7, 2, 100),  # Group B
                ]
            ),
            "feature1": np.random.normal(0, 1, n_samples),
            "feature2": np.random.normal(0, 1, n_samples),
            "predictions": np.random.beta(2, 5, n_samples),  # Skewed predictions
        }
    )


@pytest.fixture
def deterioration_feature() -> DeteriorationFeature:
    """Create a sample DeteriorationFeature."""
    return DeteriorationFeature(col="deterioration_score", threshold=6.0, label="Deterioration Score")


@pytest.fixture
def groups() -> list[Group]:
    """Create sample Group objects."""
    return [Group("Group A", ["A"], "group"), Group("Group B", ["B"], "group")]


@pytest.fixture
def mock_model() -> Mock:
    """Create a mock model with predict_proba method."""
    model = Mock()
    model.predict_proba.return_value = np.random.rand(100, 2)
    return model


@pytest.fixture
def dai_index(sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature, groups: list[Group]) -> DAIndex:
    """Create a DAIndex instance for testing."""
    return DAIndex(
        cohort=sample_cohort,
        groups=groups,
        det_feature=deterioration_feature,
        steps=10,  # Reduced for faster testing
        n_jobs=1,  # Single threaded for reproducible tests
    )


class TestDeteriorationFeature:
    """Test the DeteriorationFeature class."""

    def test_init_default_label(self) -> None:
        """Test initialization with default label."""
        df = DeteriorationFeature(col="test_col", threshold=5.0)
        assert df.col == "test_col"
        assert df.threshold == 5.0
        assert df.label == "test_col"  # Should default to col name
        assert df.func is None
        assert df.is_discrete is False
        assert df.prev_discrete_value_offset == 1
        assert df.reverse is False

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        df = DeteriorationFeature(col="score", threshold=10.0, label="Custom Score", is_discrete=True, reverse=True)
        assert df.col == "score"
        assert df.threshold == 10.0
        assert df.label == "Custom Score"
        assert df.is_discrete is True
        assert df.reverse is True

    def test_repr(self) -> None:
        """Test string representation."""
        df = DeteriorationFeature(col="test", threshold=5.0, label="Test")
        expected = "DeteriorationFeature(col='test', threshold=5.0, label='Test')"
        assert repr(df) == expected


class TestGroup:
    """Test the Group class."""

    def test_init_default_definition(self) -> None:
        """Test initialization with default definition."""
        group = Group("TestGroup")
        assert group.name == "TestGroup"
        assert group.definition == ["TestGroup"]
        assert group.col is None
        assert group.det_threshold is None

    def test_init_string_definition(self) -> None:
        """Test initialization with string definition."""
        group = Group("Male", "M", "gender")
        assert group.name == "Male"
        assert group.definition == ["M"]
        assert group.col == "gender"

    def test_init_list_definition(self) -> None:
        """Test initialization with list definition."""
        group = Group("Female", ["F", "f", "female"], "gender")
        assert group.name == "Female"
        assert group.definition == ["F", "f", "female"]
        assert group.col == "gender"

    def test_call_method(self) -> None:
        """Test calling the group to filter cohort."""
        cohort = pd.DataFrame({"gender": ["M", "F", "M", "F", "M"], "age": [25, 30, 35, 40, 45]})
        group = Group("Male", "M", "gender")
        result = group(cohort)

        assert len(result) == 3
        assert all(result["gender"] == "M")

    def test_call_method_no_col_raises_error(self) -> None:
        """Test that calling without col raises error."""
        cohort = pd.DataFrame({"test": [1, 2, 3]})
        group = Group("Test")

        with pytest.raises(ValueError, match="Group column name must be provided"):
            group(cohort)

    def test_repr(self) -> None:
        """Test string representation."""
        group = Group("Test", ["a", "b"], "col")
        expected = "Group(name='Test', col='col', definition=['a', 'b'])"
        assert repr(group) == expected


class TestDAIndexInitialization:
    """Test DAIndex initialization and setup methods."""

    def test_init_basic(
        self, sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature, groups: list[Group]
    ) -> None:
        """Test basic initialization."""
        dai = DAIndex(cohort=sample_cohort, groups=groups, det_feature=deterioration_feature)

        assert dai.cohort is sample_cohort
        assert len(dai.groups) == 2
        assert "Group A" in dai.groups
        assert "Group B" in dai.groups
        assert dai.det_feature is deterioration_feature
        assert dai.steps == 50  # default
        assert dai.model_name == "Allocation"  # default

        # Check new tracking attributes are initialized
        assert dai.group_sub_optimal_steps == {}
        assert dai.group_failed_steps == {}
        assert dai.group_step_samples == {}

    def test_init_custom_parameters(
        self, sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature, groups: list[Group]
    ) -> None:
        """Test initialization with custom parameters."""
        dai = DAIndex(
            cohort=sample_cohort,
            groups=groups,
            det_feature=deterioration_feature,
            steps=20,
            score_margin_multiplier=3.0,
            det_list_lengths=[30, 15, 8],
            model_name="Custom Model",
            decision_boundary=0.6,
        )

        assert dai.steps == 20
        assert dai.score_margin_multiplier == 3.0
        assert dai.det_list_lengths == [30, 15, 8]
        assert dai.model_name == "Custom Model"
        assert dai.decision_boundary == 0.6

    def test_setup_groups_with_group_col(
        self, sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature
    ) -> None:
        """Test setup_groups with group_col parameter."""
        groups = [Group("A"), Group("B")]
        dai = DAIndex(cohort=sample_cohort, groups=groups, det_feature=deterioration_feature, group_col="group")

        assert all(g.col == "group" for g in dai.groups.values())

    def test_setup_groups_missing_col_raises_error(
        self, sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature
    ) -> None:
        """Test that missing col in groups raises error."""
        groups = [Group("A"), Group("B")]  # No col specified

        with pytest.raises(ValueError, match="group_col must be provided"):
            DAIndex(cohort=sample_cohort, groups=groups, det_feature=deterioration_feature)


class TestDAIndexEvaluation:
    """Test DAIndex evaluation methods."""

    def test_evaluate_group_pair_by_predictions(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test evaluation by predictions column."""
        # Add predictions to cohort for both groups
        dai_index.cohort = sample_cohort

        ratios, figure = dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B"
        )

        assert isinstance(ratios, dict)
        assert "Full" in ratios
        assert "Decision" in ratios
        assert figure is not None

        # Check that step information was stored
        assert "Group A" in dai_index.group_step_samples
        assert "Group B" in dai_index.group_step_samples
        assert len(dai_index.group_step_samples["Group A"]) > 0

    def test_evaluate_group_pair_by_models(self, dai_index: DAIndex, mock_model: Mock) -> None:
        """Test evaluation by model objects."""
        feature_list = ["feature1", "feature2"]

        ratios, figure = dai_index.evaluate_group_pair_by_models(
            models=mock_model, feature_list=feature_list, reference_group="Group A", other_group="Group B"
        )

        assert isinstance(ratios, dict)
        assert "Full" in ratios
        assert "Decision" in ratios
        assert figure is not None

        # Verify model was called for both groups
        assert mock_model.predict_proba.call_count >= 2

    def test_evaluate_all_groups_by_predictions(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test evaluation of all groups by predictions."""
        dai_index.cohort = sample_cohort

        dai_index.evaluate_all_groups_by_predictions(predictions_col="predictions", reference_group="Group A")

        # Should have evaluated Group B against Group A
        assert ("Group A", "Group B") in dai_index.group_ratios
        assert ("Group A", "Group B") in dai_index.group_figures

        # Check step information was stored for both groups
        assert "Group A" in dai_index.group_step_samples
        assert "Group B" in dai_index.group_step_samples

    def test_evaluate_all_groups_by_models(self, dai_index: DAIndex, mock_model: Mock) -> None:
        """Test evaluation of all groups by models."""
        feature_list = ["feature1", "feature2"]

        dai_index.evaluate_all_groups_by_models(
            models=[mock_model], feature_list=feature_list, reference_group="Group A"
        )

        # Should have evaluated Group B against Group A
        assert ("Group A", "Group B") in dai_index.group_ratios
        assert ("Group A", "Group B") in dai_index.group_figures

    def test_invalid_group_name_raises_error(self, dai_index: DAIndex) -> None:
        """Test that invalid group names raise errors."""
        with pytest.raises(AssertionError):
            dai_index._check_group_pair("Invalid Group", "Group A")

        with pytest.raises(AssertionError):
            dai_index._check_reference_group("Invalid Group")


class TestDAIndexStepTracking:
    """Test the new step tracking functionality."""

    def test_get_group_step_samples(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting step sample information."""
        dai_index.cohort = sample_cohort

        # Run evaluation to populate step information
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B"
        )

        # Test getting step samples for a group
        step_samples = dai_index.get_group_step_samples("Group A")

        assert isinstance(step_samples, dict)
        assert len(step_samples) > 0
        assert all(isinstance(k, (int, float)) for k in step_samples.keys())
        assert all(isinstance(v, (int, np.integer)) for v in step_samples.values())

    def test_get_group_sub_optimal_steps(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting sub-optimal step information."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B"
        )

        # Test getting sub-optimal steps
        sub_optimal = dai_index.get_group_sub_optimal_steps("Group A")

        assert isinstance(sub_optimal, dict)
        # sub_optimal might be empty if all steps had optimal sample counts

    def test_get_group_failed_steps(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting failed step information."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B"
        )

        # Test getting failed steps
        failed = dai_index.get_group_failed_steps("Group A")

        assert isinstance(failed, dict)
        # failed might be empty if no steps failed

    def test_get_all_groups_step_information(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting step information for all groups."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_all_groups_by_predictions(predictions_col="predictions", reference_group="Group A")

        # Test getting information for all groups
        all_samples = dai_index.get_all_groups_step_samples()
        all_sub_optimal = dai_index.get_all_groups_sub_optimal_steps()
        all_failed = dai_index.get_all_groups_failed_steps()

        assert isinstance(all_samples, dict)
        assert isinstance(all_sub_optimal, dict)
        assert isinstance(all_failed, dict)

        # Should have information for both groups
        assert "Group A" in all_samples
        assert "Group B" in all_samples

    def test_step_tracking_error_for_unevaluated_group(self, dai_index: DAIndex) -> None:
        """Test that accessing step info for unevaluated group raises error."""
        with pytest.raises(ValueError, match="No step information available"):
            dai_index.get_group_step_samples("Group A")

        with pytest.raises(ValueError, match="No step information available"):
            dai_index.get_group_sub_optimal_steps("Group A")

        with pytest.raises(ValueError, match="No step information available"):
            dai_index.get_group_failed_steps("Group A")


class TestDAIndexResultAccess:
    """Test result access methods."""

    def test_get_group_ratio_and_figure(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting ratios and figures for specific groups."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B"
        )

        # Test getting specific results
        ratio = dai_index.get_group_ratio("Group A", "Group B")
        figure = dai_index.get_group_figure("Group A", "Group B")

        assert isinstance(ratio, dict)
        assert "Full" in ratio
        assert "Decision" in ratio
        assert figure is not None

    def test_get_all_ratios(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting all ratios as DataFrame."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_all_groups_by_predictions(predictions_col="predictions", reference_group="Group A")

        # Test getting all ratios
        all_ratios = dai_index.get_all_ratios()

        assert isinstance(all_ratios, pd.DataFrame)
        assert len(all_ratios) > 0
        assert "Full" in all_ratios.columns
        assert "Decision" in all_ratios.columns

    def test_get_all_figures(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test getting all figures."""
        dai_index.cohort = sample_cohort

        # Run evaluation
        dai_index.evaluate_all_groups_by_predictions(predictions_col="predictions", reference_group="Group A")

        # Test getting all figures
        all_figures = dai_index.get_all_figures()

        assert isinstance(all_figures, dict)
        assert len(all_figures) > 0
        assert ("Group A", "Group B") in all_figures


class TestDAIndexEdgeCases:
    """Test edge cases and error conditions."""

    def test_small_cohort_warnings(self, deterioration_feature: DeteriorationFeature, groups: list[Group]) -> None:
        """Test behavior with very small cohort that may trigger warnings."""
        small_cohort = pd.DataFrame(
            {
                "group": ["A"] * 5 + ["B"] * 5,
                "deterioration_score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "predictions": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
        )

        dai = DAIndex(
            cohort=small_cohort,
            groups=groups,
            det_feature=deterioration_feature,
            steps=5,  # Fewer steps for small cohort
            n_jobs=1,
            det_list_lengths=[3, 2, 1],  # Very small thresholds
        )

        # This should work but may generate warnings about sample sizes
        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            dai.evaluate_group_pair_by_predictions(
                predictions_col="predictions", reference_group="Group A", other_group="Group B"
            )

            # Check that step information was still collected
            assert "Group A" in dai.group_step_samples
            assert "Group B" in dai.group_step_samples

    def test_rerun_parameter(self, dai_index: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test the rerun parameter functionality."""
        dai_index.cohort = sample_cohort

        # First evaluation
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B", rerun=True
        )

        # Store original step samples
        original_samples = dai_index.get_group_step_samples("Group A").copy()

        # Second evaluation with rerun=False should use cached results
        dai_index.evaluate_group_pair_by_predictions(
            predictions_col="predictions", reference_group="Group A", other_group="Group B", rerun=False
        )

        # Step samples should be the same (cached)
        current_samples = dai_index.get_group_step_samples("Group A")
        assert original_samples == current_samples


class TestDAIndexIntegration:
    """Integration tests with model-like objects."""

    def test_with_mock_sklearn_model(
        self, sample_cohort: pd.DataFrame, deterioration_feature: DeteriorationFeature, groups: list[Group]
    ) -> None:
        """Test with a mock sklearn-like model."""

        # Create a mock model that behaves like sklearn
        class MockSklearnModel:
            def __init__(self) -> None:
                np.random.seed(42)

            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                # Return realistic probabilities based on input features
                n_samples = x.shape[0]
                # Generate probabilities that sum to 1 for each row
                prob_positive = np.random.beta(2, 3, n_samples)  # Skewed towards 0
                prob_negative = 1 - prob_positive
                return np.column_stack([prob_negative, prob_positive])

        model = MockSklearnModel()

        dai = DAIndex(
            cohort=sample_cohort,
            groups=groups,
            det_feature=deterioration_feature,
            steps=5,  # Fewer steps for faster testing
            n_jobs=1,
        )

        # Test evaluation with mock model
        ratios, figure = dai.evaluate_group_pair_by_models(
            models=model, feature_list=["feature1", "feature2"], reference_group="Group A", other_group="Group B"
        )

        assert isinstance(ratios, dict)
        assert "Full" in ratios
        assert "Decision" in ratios
        assert figure is not None

        # Test step tracking worked
        step_samples = dai.get_group_step_samples("Group A")
        assert len(step_samples) > 0


if __name__ == "__main__":
    pytest.main([__file__])
