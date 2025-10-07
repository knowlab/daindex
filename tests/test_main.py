"""
Test suite for the main daindex module.

This test file provides comprehensive coverage for the main classes and functions
in daindex/main.py while maintaining minimal setup requirements.
"""

import warnings
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KernelDensity

from daindex.main import DAIndex, DeteriorationFeature, Group, ProbabilisticModel


class TestDeteriorationFeature:
    """Test cases for DeteriorationFeature class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        feature = DeteriorationFeature(col="test_col", threshold=0.5)
        assert feature.col == "test_col"
        assert feature.threshold == 0.5
        assert feature.label == "test_col"  # defaults to col
        assert feature.func is None
        assert feature.is_discrete is False
        assert feature.reverse is False

    def test_init_with_custom_params(self) -> None:
        """Test initialization with custom parameters."""

        def func(x: float) -> float:
            return x * 2

        feature = DeteriorationFeature(
            col="custom_col",
            threshold=0.8,
            label="Custom Label",
            func=func,
            is_discrete=True,
            reverse=True,
        )
        assert feature.col == "custom_col"
        assert feature.threshold == 0.8
        assert feature.label == "Custom Label"
        assert feature.func == func
        assert feature.is_discrete is True
        assert feature.reverse is True


class TestGroup:
    """Test cases for Group class."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        group = Group("Test Group")
        assert group.name == "Test Group"
        assert group.definition == ["Test Group"]  # defaults to [name]
        assert group.col is None

    def test_init_single_definition_to_list(self) -> None:
        """Test that single definition value is converted to list."""
        group = Group("Test", definition="single_value")
        assert group.definition == ["single_value"]

    def test_call_basic_filtering(self) -> None:
        """Test basic group filtering functionality."""
        cohort = pd.DataFrame({"group_col": ["A", "B", "A", "C"], "value": [1, 2, 3, 4]})
        group = Group("Group A", definition="A", col="group_col")
        result = group(cohort)
        expected = pd.DataFrame({"group_col": ["A", "A"], "value": [1, 3]}, index=[0, 2])
        pd.testing.assert_frame_equal(result, expected)

    def test_call_multiple_definitions(self) -> None:
        """Test filtering with multiple definition values."""
        cohort = pd.DataFrame({"sex": ["M", "F", "m", "F", "f", "female"], "value": [1, 2, 3, 4, 5, 6]})
        group = Group("Female", definition=["F", "f", "female"], col="sex")
        result = group(cohort)
        expected = pd.DataFrame({"sex": ["F", "F", "f", "female"], "value": [2, 4, 5, 6]}, index=[1, 3, 4, 5])
        pd.testing.assert_frame_equal(result, expected)

    def test_call_without_col_raises_error(self) -> None:
        """Test that calling without col raises ValueError."""
        cohort = pd.DataFrame({"value": [1, 2, 3]})
        group = Group("Test")  # no col specified
        with pytest.raises(ValueError, match="Group column name must be provided"):
            group(cohort)


class TestDAIndex:
    """Test cases for DAIndex class."""

    @pytest.fixture
    def sample_cohort(self) -> pd.DataFrame:
        """Create a sample cohort DataFrame for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": [25, 35, 45, 55, 65, 75, 85, 95],
                "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
                "feature_1": [0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.85, 0.95],
                "feature_2": [1, 0, 1, 0, 1, 0, 1, 0],
                "deterioration": [0.2, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0],
                "outcome": [0, 0, 0, 1, 1, 1, 1, 1],
                "predictions": [0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95],
            }
        )

    @pytest.fixture
    def sample_groups(self) -> list[Group]:
        """Create sample groups for testing."""
        return [
            Group(name="Male", definition="M", col="sex"),
            Group(name="Female", definition="F", col="sex"),
        ]

    @pytest.fixture
    def sample_det_feature(self) -> DeteriorationFeature:
        """Create sample deterioration feature for testing."""
        return DeteriorationFeature(col="deterioration", threshold=0.5)

    @pytest.fixture
    def dai_instance(
        self, sample_cohort: pd.DataFrame, sample_groups: list[Group], sample_det_feature: DeteriorationFeature
    ) -> DAIndex:
        """Create a DAIndex instance for testing."""
        return DAIndex(
            cohort=sample_cohort,
            groups=sample_groups,
            det_feature=sample_det_feature,
            n_bins=10,  # smaller for faster tests
        )

    def test_init(
        self, sample_cohort: pd.DataFrame, sample_groups: list[Group], sample_det_feature: DeteriorationFeature
    ) -> None:
        """Test DAIndex initialization."""
        dai = DAIndex(
            cohort=sample_cohort,
            groups=sample_groups,
            det_feature=sample_det_feature,
        )
        assert dai.cohort is sample_cohort
        assert len(dai.groups) == 2
        assert "Male" in dai.groups
        assert "Female" in dai.groups
        assert dai.det_feature is sample_det_feature
        assert dai.model_name == "Allocation"
        assert dai.decision_boundary == 0.5

    def test_setup_groups_missing_col_raises_error(
        self, sample_cohort: pd.DataFrame, sample_det_feature: DeteriorationFeature
    ) -> None:
        """Test that missing col in groups raises ValueError."""
        groups = [Group("Group1", "A")]  # no col specified
        dai = DAIndex.__new__(DAIndex)
        dai.cohort = sample_cohort
        with pytest.raises(ValueError, match="group_col must be provided"):
            dai.setup_groups(groups)

    def test_kde_estimate(self, dai_instance: DAIndex) -> None:
        """Test KDE estimation."""
        data = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
        kde = dai_instance._kde_estimate(data)
        assert isinstance(kde, KernelDensity)

    def test_stepped_severity(self, dai_instance: DAIndex) -> None:
        """Test stepped severity calculation."""
        probs = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
        result = dai_instance._stepped_severity(probs, 0.4, 0.6, 0.0, 0.1, 0.0)
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_obtain_da_index_insufficient_samples(self, dai_instance: DAIndex) -> None:
        """Test DA index calculation with insufficient samples."""
        # Mock group scores with very few matching samples
        dai_instance.group_scores = {"Male": [0.1, 0.9, 0.9, 0.9]}  # only first matches bin
        length, di_ret, sub_opt, failed = dai_instance._obtain_da_index("Male", (0.05, 0.15))
        assert failed is True
        assert di_ret == 0.0

    def test_get_scores_from_models(self, dai_instance: DAIndex, sample_cohort: pd.DataFrame) -> None:
        """Test score calculation from models."""
        # Create a mock model
        mock_model = MagicMock(spec=ProbabilisticModel)
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])

        scores = dai_instance._get_scores_from_models("Male", [mock_model], ["feature_1", "feature_2"])
        assert len(scores) == 4  # 4 males in sample data
        assert all(0 <= score <= 1 for score in scores)

    def test_area_under_curve(self, dai_instance: DAIndex) -> None:
        """Test area under curve calculation."""
        # Simple test data
        w_data = np.array([[0.0, 0.1], [0.25, 0.3], [0.5, 0.6], [0.75, 0.4], [1.0, 0.2]])
        area, decision_area = dai_instance._area_under_curve(w_data)
        assert isinstance(area, float)
        assert isinstance(decision_area, float)
        assert area > 0
        assert decision_area >= 0

    def test_calculate_group_curve(self, dai_instance: DAIndex) -> None:
        """Test group curve calculation."""
        # Mock k-steps data
        dai_instance.group_ksteps = {"Male": np.array([[0.1, 5, 0.3], [0.5, 3, 0.7], [0.9, 2, 0.9]])}
        curve = dai_instance._calculate_group_curve("Male")
        assert "x" in curve
        assert "y" in curve
        assert "sample_counts" in curve
        assert "det_threshold" in curve

    def test_calculate_group_ratio(self, dai_instance: DAIndex) -> None:
        """Test group ratio calculation."""
        # Mock curve data
        dai_instance.group_curves = {
            "Male": {"x": np.array([0.1, 0.5, 0.9]), "y": np.array([0.3, 0.6, 0.9])},
            "Female": {"x": np.array([0.2, 0.6, 0.8]), "y": np.array([0.4, 0.7, 0.8])},
        }
        ratio = dai_instance._calculate_group_ratio("Male", "Female")
        assert "Full" in ratio
        assert "Decision" in ratio

    def test_check_group_invalid(self, dai_instance: DAIndex) -> None:
        """Test group validation with invalid group."""
        with pytest.raises(KeyError, match="Invalid group name"):
            dai_instance._check_group("InvalidGroup")

    def test_validate_group_inputs_none(self, dai_instance: DAIndex) -> None:
        """Test group input validation with None (all groups)."""
        result = dai_instance._validate_group_inputs(None)
        assert set(result) == {"Male", "Female"}

    def test_evaluate_groups_by_predictions(self, dai_instance: DAIndex) -> None:
        """Test evaluation using pre-calculated predictions."""
        with (
            patch.object(dai_instance, "_get_group_ksteps") as mock_ksteps,
            patch.object(dai_instance, "_calculate_group_curve") as mock_curve,
        ):
            mock_ksteps.return_value = np.array([[0.1, 5, 0.3]])
            mock_curve.return_value = {"x": np.array([0.1]), "y": np.array([0.3])}

            dai_instance.evaluate_groups_by_predictions(predictions_col="predictions", reference_group="Male")

            assert "Male" in dai_instance.group_scores
            assert "Female" in dai_instance.group_scores

    def test_evaluate_groups_by_models(self, dai_instance: DAIndex) -> None:
        """Test evaluation using model objects."""
        mock_model = MagicMock(spec=ProbabilisticModel)
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]])

        with (
            patch.object(dai_instance, "_get_group_ksteps") as mock_ksteps,
            patch.object(dai_instance, "_calculate_group_curve") as mock_curve,
        ):
            mock_ksteps.return_value = np.array([[0.1, 5, 0.3]])
            mock_curve.return_value = {"x": np.array([0.1]), "y": np.array([0.3])}

            dai_instance.evaluate_groups_by_models(
                models=mock_model, feature_list=["feature_1", "feature_2"], reference_group="Male"
            )

    def test_get_plots(self, dai_instance: DAIndex) -> None:
        """Test plot generation."""
        # Mock the required data
        dai_instance.group_curves = {
            "Male": {"x": np.array([0.1, 0.5, 0.9]), "y": np.array([0.3, 0.6, 0.9])},
            "Female": {"x": np.array([0.2, 0.6, 0.8]), "y": np.array([0.4, 0.7, 0.8])},
        }
        dai_instance.group_ratios = {("Male", "Female"): {"Full": 0.1, "Decision": 0.05}}

        fig = dai_instance.get_plots(reference_group="Male")
        assert isinstance(fig, plt.Figure)

    def test_get_ratios(self, dai_instance: DAIndex) -> None:
        """Test getting ratios DataFrame."""
        dai_instance.group_ratios = {
            ("Male", "Female"): {"Full": 0.1, "Decision": 0.05},
            ("Female", "Male"): {"Full": -0.09, "Decision": -0.047},
        }
        df = dai_instance.get_ratios()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_get_curves_single_group(self, dai_instance: DAIndex) -> None:
        """Test getting curve data for single group."""
        dai_instance.group_curves = {
            "Male": {"x": np.array([0.1]), "y": np.array([0.3])},
        }
        result = dai_instance.get_curves("Male")
        assert "x" in result
        assert "y" in result

    def test_warning_on_sub_optimal_bins(self, dai_instance: DAIndex) -> None:
        """Test that warnings are raised for sub-optimal bins."""
        dai_instance.group_scores = {"Male": [0.1, 0.2, 0.3]}  # minimal data

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This should trigger sub-optimal sample warnings
            with patch.object(dai_instance, "_obtain_da_index", return_value=(15, 0.5, True, False)):
                dai_instance._get_group_ksteps("Male")

            # Check that a warning was issued
            assert len(w) > 0
            assert "sub-optimal" in str(w[0].message)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_end_to_end_workflow_with_predictions(self) -> None:
        """Test complete workflow using predictions."""
        # Create test data
        cohort = pd.DataFrame(
            {
                "group": ["A", "B"] * 20,
                "feature1": np.random.random(40),
                "feature2": np.random.random(40),
                "deterioration": np.random.random(40),
                "outcome": np.random.choice([0, 1], 40),
                "predictions": np.random.random(40),
            }
        )

        groups = [
            Group("Group A", "A", "group"),
            Group("Group B", "B", "group"),
        ]

        det_feature = DeteriorationFeature("deterioration", 0.5)

        dai = DAIndex(cohort, groups, det_feature, n_bins=5)

        # This should complete without errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sample size warnings for test data
            dai.evaluate_groups_by_predictions("predictions", "Group A")

        # Verify results exist
        assert "Group A" in dai.group_scores
        assert "Group B" in dai.group_scores

    def test_end_to_end_workflow_with_models(self) -> None:
        """Test complete workflow using model objects."""
        # Create test data
        np.random.seed(42)
        cohort = pd.DataFrame(
            {
                "group": ["A", "B"] * 20,
                "feature1": np.random.random(40),
                "feature2": np.random.random(40),
                "deterioration": np.random.random(40),
                "outcome": np.random.choice([0, 1], 40),
            }
        )

        groups = [
            Group("Group A", "A", "group"),
            Group("Group B", "B", "group"),
        ]

        det_feature = DeteriorationFeature("deterioration", 0.5)

        # Train a simple model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(cohort[["feature1", "feature2"]], cohort["outcome"])

        dai = DAIndex(cohort, groups, det_feature, n_bins=5)

        # This should complete without errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore sample size warnings
            dai.evaluate_groups_by_models(model, ["feature1", "feature2"], "Group A")

        # Verify results exist
        assert "Group A" in dai.group_scores
        assert "Group B" in dai.group_scores

        # Test getting results
        ratios_df = dai.get_ratios()
        assert isinstance(ratios_df, pd.DataFrame)

        curves = dai.get_curves()
        assert "Group A" in curves
        assert "Group B" in curves
