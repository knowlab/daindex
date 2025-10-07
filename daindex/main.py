import itertools
import warnings
from collections import namedtuple
from typing import Callable, Literal, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from tqdm.autonotebook import tqdm

BinResult = namedtuple("BinResult", ["score", "length", "di_ret", "sub_opt", "failed"])


@runtime_checkable
class ProbabilisticModel(Protocol):
    def predict_proba(self, in_matrix: np.ndarray) -> np.ndarray: ...


class DeteriorationFeature(object):
    """
    Class to define the feature upon which to calculate the DA Index.
    This is passed into the `DAIndex` class.

    Args:
        col: The column name in the cohort DataFrame that contains the feature.
        threshold: The threshold value for the deterioration index.
        label: The label for the deterioration index.
        func: An optional function to apply to each row to extract the feature value.
        is_discrete: Whether the feature is discrete.
        prev_discrete_value_offset: The difference between the threshold and the previous legitimate value.
        reverse: For calculating p(in_matrix<threshold), i.e., the smaller the measure value the more severe a patient.
    """

    def __init__(
        self,
        col: str,
        threshold: float,
        label: str = None,
        func: Callable = None,
        is_discrete: bool = False,
        prev_discrete_value_offset: int = 1,
        reverse: bool = False,
    ) -> None:
        self.col = col
        self.threshold = threshold
        self.label = label or col
        self.func = func
        self.is_discrete = is_discrete
        self.prev_discrete_value_offset = prev_discrete_value_offset
        self.reverse = reverse

    def __repr__(self) -> str:
        return f"DeteriorationFeature(col='{self.col}', threshold={self.threshold}, label='{self.label}')"


class Group(object):
    """
    Class to define a sub-group within the cohort upon which to calculate the DA Index.
    A list of these should be passed into the `groups` parameter of the `DAIndex` class.

    Args:
        name: The name of the group in the desired formatting to display on plots and use as a key to extract results
            etc.
        definition: The value(s) in the column that define the group.
            E.g. could be a list of values, a single value, string, number, etc.
            If not provided, defaults to the name.
        col: The column name in the cohort DataFrame that contains the group definition. Can be overriden by the
            `group_col` parameter of the `DAIndex` class.
        det_threshold: The threshold value for the deterioration index, overriding the `threshold` attribute of the
            `DeteriorationFeature`.
        get_group: This is an optional argument to allow passing in of a more complex function
            that returns the group DataFrame.
            If not provided, the group DataFrame is obtained by filtering the cohort DataFrame
            by the `col` and `definition`.

    Methods:
        Call the object to return the group DataFrame by operating on the cohort DataFrame.

    Examples:
        >>> cohort = pd.DataFrame({"group_col": ["group_1", "group_2", "group_1", "group_3"]})
        >>> group = Group("Group 1", "group_1", "group_col")
        >>> group(cohort)

          group_col
        0   group_1
        2   group_1

        >>> cohort = pd.DataFrame({"sex": ["M", "F", "m", "F", "f", "female"]})
        >>> group = Group("Female", ["F", "f", "female"], "sex")
        >>> group(cohort)

               sex
        1   F
        3   F
        4   f
        5   female
    """

    def __init__(
        self,
        name: str,
        definition: list | str | None = None,
        col: str | None = None,
        det_threshold: float | None = None,
        get_group: Callable | None = None,
    ) -> None:
        self.name = name
        if definition is None:
            self.definition = [name]
        elif not isinstance(definition, list):
            self.definition = [definition]
        else:
            self.definition = definition
        self.col = col
        self.det_threshold = det_threshold
        self._get_group = get_group

    def __repr__(self) -> str:
        return f"Group(name='{self.name}', col='{self.col}', definition={self.definition})"

    def __call__(self, cohort: pd.DataFrame) -> pd.DataFrame:
        if not self.col:
            raise ValueError("Group column name must be provided")
        return (
            self._get_group(self, cohort)
            if self._get_group is not None
            else cohort[cohort[self.col].isin(self.definition)]
        )


class DAIndex(object):
    """
    Class to calculate the Deterioration Allocation Index (DAI) for a cohort of patients.
    The DAI is a measure of the deterioration of a patient's health over time, based on a given feature.
    We compare the DAI between two groups relative to a given model's predictions. This then gives us a measure of how
    fair the model's predictions are across the two groups.

    1. This class should be instantiated first with a `DeteriorationFeature` object and a list of `Group` objects.
    2. The `evaluate_groups_by_predictions` or `evaluate_groups_by_models` methods can then be called to calculate
    the DAI for a reference and list of groups.
    3. The results can be accessed using the `get_plot`, `get_ratios`, and `get_curves` methods.
    4. Any issues during the calculations are raised as warnings, and can be accessed using the `get_sub_optimal_bins`,
    `get_failed_bins`, and `get_bin_samples` methods.

    All of the functions in 3 can be called without arguments to return results for all groups, or with a single group /
    list of groups to return results for those groups only.

    Args:
        cohort: The DataFrame containing the data for the cohort.
        groups: A list of `Group` objects representing the sub-groups within the cohort.
        det_feature: A `DeteriorationFeature` object representing the feature upon which to calculate the DAI.
        group_col: Optional column name in the cohort DataFrame that contains the group definition.
            Specifying this overrides the `col` attribute of the `Group` objects.
        n_bins: The number of bins to use for the DAI calculation.
        acceptable_samples: The acceptable number of samples to use for each bin's DAI calculation.
        minimum_samples: The minimum number of samples to use for each bin's DAI calculation.
        score_margin_multiplier: The multiplier to use for the score margin.
        bandwidth: The bandwidth to use for the KDE.
        optimise_bandwidth: Whether to search for the optimal bandwidth.
        kernel: The kernel to use for the KDE.
        n_samples: The number of samples to use for the KDE.
        weight_sum_steps: The number of steps for the weighted sum of k-step cutoffs.
        n_jobs: The number of jobs to run in parallel.
        model_name: The name of the model to use in the plots.
        decision_boundary: The decision boundary for the DA curve. Defaults to 0.5.

    Methods:
        setup_groups: Set up the groups for the DAI calculation.
        setup_daauc_params: Set up the parameters for the DAI calculation.
        setup_deterioration_feature: Set up the deterioration feature for the DAI calculation.
        evaluate_groups_by_predictions: Calculate the DAI for group(s) based on pre-calculated model predictions.
        evaluate_groups_by_models: Calculate the DAI for group(s) based on a list of trained models.
        get_plot: Get the DAI plots for the (specified) group(s).
        get_ratios: Get the DAI ratios for the (specified) group(s).
        get_curves: Get the curve data for the (specified) group(s).
        get_sub_optimal_bins: Get information about any sub-optimal bins for the (specified) group(s).
        get_failed_bins: Get information about any failed bins for the (specified) group(s).
        get_bin_samples: Get the number of samples used for each bin for the (specified) group(s).

    Raises:
        ValueError: If the group names provided to any method are not valid.
        UserWarning: If there are any issues during the DAI calculations, such as sub-optimal or insufficient samples.

    Examples:
        >>> import pandas as pd
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from daindex import DAIndex, DeteriorationFeature, Group
        >>> cohort = pd.DataFrame(
        ...     {
        ...         "age": [25, 35, 45, 55, 65, 75, 85, 95],
        ...         "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        ...         "feature_1": [0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.85, 0.95],
        ...         "feature_2": [1, 0, 1, 0, 1, 0, 1, 0],
        ...         "deterioration": [0.2, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0],
        ...         "outcome": [0, 0, 0, 1, 1, 1, 1, 1],
        ...     }
        ... )
        >>> det_feature = DeteriorationFeature(col="deterioration", threshold=0.5)
        >>> group_1 = Group(name="Male", definition="M", col="sex")
        >>> group_2 = Group(name="Female", definition="F", col="sex")
        >>> dai = DAIndex(cohort, groups=[group_1, group_2], det_feature=det_feature)
        >>> model = RandomForestClassifier()
        >>> model.fit(cohort[["feature_1", "feature_2"]], cohort["outcome"])
        >>> dai.evaluate_groups_by_models(model, feature_list=["feature_1", "feature_2"], reference_group="Male")
        >>> dai.get_plot(reference_group="Male")
    """

    def setup_groups(self, groups: list[Group], group_col: str = None) -> None:
        if group_col is not None:
            groups = [Group(g.name, g.definition, group_col, g.det_threshold, g._get_group) for g in groups]
        elif any(g.col is None for g in groups):
            raise ValueError("group_col must be provided if any group objects do not have a col attribute")
        self.groups = {g.name: g for g in groups}

    def setup_daauc_params(
        self,
        n_bins: int = 50,
        acceptable_samples: int = 20,
        minimum_samples: int = 5,
        score_margin_multiplier: float = 2.0,
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        optimise_bandwidth: bool = False,
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        n_samples: int = 10000,
        weight_sum_steps: int = 50,
        n_jobs: int = -1,
    ) -> None:
        self.acceptable_samples = acceptable_samples
        self.minimum_samples = minimum_samples
        self.score_margin_multiplier = score_margin_multiplier
        self.bins = {
            s: (
                max(0.0, s - (1 / (2 * n_bins)) * self.score_margin_multiplier),
                min(1.0, s + (1 / (2 * n_bins)) * self.score_margin_multiplier),
            )
            for s in np.linspace(0, 1, n_bins + 1)[1:] - 1 / (2 * n_bins)
        }
        self.bandwidth = bandwidth
        self.optimise_bandwidth = optimise_bandwidth
        self.kernel = kernel
        self.n_samples = n_samples
        self.weight_sum_steps = weight_sum_steps
        self.n_jobs = n_jobs

    def setup_deterioration_feature(self, det_feature: DeteriorationFeature) -> None:
        self.det_feature = det_feature
        self.min_det_val = min(g(self.cohort)[self.det_feature.col].min() for g in self.groups.values())
        self.max_det_val = max(g(self.cohort)[self.det_feature.col].max() for g in self.groups.values())

    def __init__(
        self,
        cohort: pd.DataFrame,
        groups: list[Group],
        det_feature: DeteriorationFeature,
        group_col: str = None,
        n_bins: int = 50,
        acceptable_samples: int = 20,
        minimum_samples: int = 5,
        score_margin_multiplier: float = 2.0,
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        optimise_bandwidth: bool = False,
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        n_samples: int = 10000,
        weight_sum_steps: int = 50,
        n_jobs: int = -1,
        model_name: str = "Allocation",
        decision_boundary: float = 0.5,
    ) -> None:
        self.cohort = cohort
        self.setup_groups(groups, group_col)
        self.setup_deterioration_feature(det_feature)
        self.setup_daauc_params(
            n_bins,
            acceptable_samples,
            minimum_samples,
            score_margin_multiplier,
            bandwidth,
            optimise_bandwidth,
            kernel,
            n_samples,
            weight_sum_steps,
            n_jobs,
        )

        self.model_name = model_name
        self.decision_boundary = decision_boundary

        self.group_scores = {}
        self.group_ksteps = {}
        self.group_curves = {}  # Stores curve data for each group
        self.group_ratios = {}  # Stores ratios between each pair of groups
        self.group_bin_samples = {}  # Stores sample counts for each bin for each group
        self.group_sub_optimal_bins = {}  # Stores sub-optimal bins for each group
        self.group_failed_bins = {}  # Stores failed bins for each group

    def _gridsearch_bandwidth(self, in_matrix: np.ndarray) -> float:
        """
        Search for the best bandwith for the KDE
        """
        bandwidths = np.linspace(0, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=KFold(5))
        grid.fit(in_matrix)
        return grid.best_params_["bandwidth"]

    def _kde_estimate(self, in_matrix: np.ndarray) -> KernelDensity:
        """
        Kernel density estimation to get probability
        """
        if self.optimise_bandwidth:
            bandwidth = self._gridsearch_bandwidth(in_matrix)
        else:
            bandwidth = self.bandwidth
        kde = KernelDensity(bandwidth=bandwidth, kernel=self.kernel)
        kde.fit(in_matrix)

        # detect pulse like PDF
        if bandwidth < 0.1:
            # force is_discrete because the bestfitted_bandwith_bw would lead to one
            self.det_feature.is_discrete = True
        elif bandwidth > 0.7:
            # force is_discrete to be false because the fitted_bandwith would lead to one
            self.det_feature.is_discrete = False

        return kde

    def _search_for_zero_mass_index(self, kde: KernelDensity, min_v: float, n_samples: int = 100) -> tuple[int, float]:
        """
        Search near zero probability mass for boundary adjustment
        """
        bins = np.linspace(min_v - 10, min_v, n_samples)
        pd_vals = np.exp(kde.score_samples(bins.reshape(-1, 1)))
        first_zero_idx = np.max(np.where(pd_vals < 1e-10))
        return first_zero_idx, bins[first_zero_idx]

    def _get_threshold_index(
        self,
        threshold: float,
        low_bound: float,
        step: float,
        boundary_offset: float,
    ) -> int:
        """
        Calculate threshold value index
        """
        threshold_index = int((threshold - low_bound) / step)
        if threshold == low_bound + boundary_offset:
            threshold_index = int((threshold - low_bound - boundary_offset) / step)
        elif self.det_feature.is_discrete:
            # discrete values will lead to PDF shape like pulses,
            # it's important to start with the valley between the pulse you want to include and the one before that
            # the following is to find the valley index
            threshold_index_prev = int((threshold - self.det_feature.prev_discrete_value_offset - low_bound) / step)
            threshold_index = int(threshold_index_prev + (threshold_index - threshold_index_prev) / 2)
        return threshold_index

    def _stepped_severity(
        self,
        probs: np.ndarray,
        s: float,
        e: float,
        low_bound: float,
        step_width: float,
        boundary_offset: float,
    ) -> float:
        """
        To quantify severity by considering higher values as more severe.
        This is done by weighted sum by integrating probs from a threshold - s
        """

        bins = np.linspace(s, e, self.weight_sum_steps)
        bin_probs = []
        for i, t in enumerate(bins):
            idx1 = self._get_threshold_index(t, low_bound, step_width, boundary_offset)
            if i < len(bins) - 1:
                idx2 = self._get_threshold_index(bins[i + 1], low_bound, step_width, boundary_offset)
                bin_probs.append(probs[idx1:idx2].sum())
            else:
                (
                    bin_probs.append(probs[idx1:].sum())
                    if not self.det_feature.reverse
                    else bin_probs.append(
                        probs[idx1 : self._get_threshold_index(e, low_bound, step_width, boundary_offset)].sum()
                    )
                )
        s = 0

        def weight_function(x: float) -> float:
            return x + 1

        w = 0
        for i, p in enumerate(bin_probs):
            if not self.det_feature.reverse:
                s += weight_function(i) * p
            else:
                s += weight_function(len(bin_probs) - i) * p
            w += weight_function(i)
        return s / w

    def _deterioration_index(self, in_matrix: np.ndarray, group: str) -> float:
        """
        Obtain deterioration index.

        Args:
            in_matrix (np.ndarray): The input matrix containing the feature values for the group.
            group (str): The name of the group for which to calculate the deterioration index.

        Returns:
            float: The deterioration index for the group.
        """

        # estimate density function
        kde = self._kde_estimate(in_matrix)

        # automatically adjust on boundaries
        _, adjusted_min = self._search_for_zero_mass_index(kde, self.min_det_val)
        boundary_offset = self.min_det_val - adjusted_min

        low_bound = self.min_det_val - boundary_offset
        up_bound = self.max_det_val + boundary_offset

        # use learned KDE estimator to get probability
        bins = np.linspace(low_bound, up_bound, self.n_samples)
        kd_vals = kde.score_samples(bins.reshape(-1, 1))  # Get PDF values for each x
        step_width = (up_bound - low_bound) / self.n_samples  # get the step
        probs = np.exp(kd_vals) * step_width  # get the approximate prob at each point using the integral of the PDF

        # severity quantification
        det_threshold = self.groups[group].det_threshold or self.det_feature.threshold
        if self.det_feature.reverse:
            s = low_bound
            e = min(det_threshold, up_bound)
        else:
            s = max(det_threshold, low_bound)
            e = up_bound

        # stepped quantification that considers higher/lower the value, more severe the patients are
        sqs = self._stepped_severity(probs, s, e, low_bound, step_width, boundary_offset)

        return round(sqs, 6)

    def _obtain_da_index(self, group: str, bin_bounds: tuple[float, float]) -> tuple[int, float, bool, bool]:
        """
        Calculates the Deterioration Allocation Index (DAI) for a given cohort.

        Args:
            group: The name of the group for which to calculate the DAI.
            bin_bounds: A tuple containing the lower and upper bounds of the score range to consider.

        Returns:
            A tuple containing:
                - int: The length of the det_list.
                - float: The k-step value from the deterioration index calculation.
                - bool: Whether the number of samples is sub-optimal.
                - bool: Whether the calculation failed due to insufficient samples.

        Raises:
            UserWarning: If the number of samples is sub-optimal or insufficient for DAI calculation.
        """
        if group not in self.group_scores:
            raise ValueError(f"Scores for group '{group}' not found. Please run evaluation first.")

        lb, ub = bin_bounds

        det_list = []
        i = 0
        sub_opt = False

        df = self.groups[group](self.cohort)
        for _, r in df.iterrows():
            p = self.group_scores[group][i]
            if lb <= p <= ub:
                if self.det_feature.func is not None:
                    det_list.append(self.det_feature.func(r))
                else:
                    det_list.append(r[self.det_feature.col])
            i += 1

        if len(det_list) < self.minimum_samples:
            return len(det_list), 0.0, False, True
        elif len(det_list) < self.acceptable_samples:
            sub_opt = True

        in_matrix = np.array(det_list)
        di_ret = self._deterioration_index(in_matrix[~np.isnan(in_matrix)].reshape(-1, 1), group)
        return len(det_list), di_ret, sub_opt, False

    def _get_group_ksteps(self, group: str) -> np.ndarray:
        def process_bin(s: float, bounds: tuple[float, float]) -> tuple[float, int, float, bool, bool]:
            length, di_ret, sub_opt, failed = self._obtain_da_index(group, bounds)
            return BinResult(s, length, di_ret, sub_opt, failed)

        ret = Parallel(n_jobs=self.n_jobs)(
            delayed(process_bin)(s, bounds)
            for s, bounds in tqdm(
                self.bins.items(), desc=f"Calculating k-steps for '{group}' group", position=1, leave=False
            )
        )

        # Store step information internally
        self.group_sub_optimal_bins[group] = {s.score: s.length for s in ret if s.sub_opt}
        self.group_failed_bins[group] = {s.score: s.length for s in ret if s.failed}
        self.group_bin_samples[group] = {s.score: s.length for s in ret}

        sub_opt_list = ", ".join([f"{s.score}: {s.length}" for s in ret if s.sub_opt])
        failed_list = ", ".join([f"{s.score}: {s.length}" for s in ret if s.failed])
        if sub_opt_list or failed_list:
            message = f"\nIssues were encountered during the {group} group calculation:"
            if sub_opt_list:
                message += f"\nThere are a sub-optimal number of samples for these bin scores: {sub_opt_list}"
            if failed_list:
                message += f"\nThere are too few samples for these bin scores: {failed_list}"
            warnings.warn(message, stacklevel=2)
        return np.array([(s.score, s.length, s.di_ret) for s in ret if not s.failed])

    def _get_scores_from_models(
        self, group: str, models: list[ProbabilisticModel], feature_list: list[str]
    ) -> np.ndarray:
        """
        Computes the mean predicted probabilities for a list of models.

        Args:
            group: The group name for which to compute the predicted probabilities
            models: A list of trained model objects that have a `predict_proba` method.
            feature_list: A list of column names in `df` to be used as features for prediction.

        Returns:
            An array of mean predicted probabilities for the positive class.
        """
        predicted_probs = np.array(
            [m.predict_proba(self.groups[group](self.cohort)[feature_list].to_numpy()) for m in models]
        )
        return predicted_probs[:, :, 1].mean(axis=0)

    def _area_under_curve(self, w_data: np.ndarray) -> tuple[float, float]:
        """
        Calculate total area under curve and decision area (above decision_boundary).
        Uses linear interpolation to include a point exactly at the decision_boundary
        if it falls between two x-values.
        """
        area = 0.0
        decision_area = 0.0
        decision_boundary = self.decision_boundary

        # Sort by x just in case
        w_data = w_data[np.argsort(w_data[:, 0])]

        # Optionally insert an interpolated point at decision_boundary
        x_vals = w_data[:, 0]
        if decision_boundary > x_vals.min() and decision_boundary < x_vals.max():
            # find where boundary falls between two points
            idx = np.searchsorted(x_vals, decision_boundary)
            if idx < len(w_data) and x_vals[idx] != decision_boundary:
                # interpolate y at decision_boundary
                x0, y0 = w_data[idx - 1, 0], w_data[idx - 1, 1]
                x1, y1 = w_data[idx, 0], w_data[idx, 1]
                t = (decision_boundary - x0) / (x1 - x0)
                y_interp = y0 + (y1 - y0) * t
                # insert new point
                interp_point = np.array([[decision_boundary, y_interp]])
                w_data = np.insert(w_data, idx, interp_point, axis=0)

        # Integrate trapezoids
        for i in range(1, len(w_data)):
            x0, y0 = w_data[i - 1]
            x1, y1 = w_data[i]
            trap_area = (y0 + y1) * (x1 - x0) / 2  # * w_data[i, 2]  # weight by sample count
            area += trap_area
            if x1 > decision_boundary:
                # Handle partial trapezoid if it starts before boundary
                if x0 < decision_boundary:
                    # area from boundary to x1
                    # Already have interpolated point, so this is exact
                    pass  # covered by insertion
                decision_area += trap_area

        return area, decision_area

    def _calculate_group_curve(self, group: str) -> dict[str, float | np.ndarray]:
        """
        Calculate curve data for a single group without plotting.

        Args:
            group: The name of the group to calculate the curve for.

        Returns:
            A dictionary containing curve data, areas, and metadata.
        """
        if group not in self.group_ksteps:
            raise ValueError(f"K-steps for group '{group}' not found. Please run evaluation first.")

        data = self.group_ksteps[group]
        w_data = data[np.where(data[:, 1] > 0)][:, [0, 2, 1]]

        return {
            "x": w_data[:, 0],
            "y": w_data[:, 1],
            "sample_counts": w_data[:, 2],
            "det_threshold": self.groups[group].det_threshold or self.det_feature.threshold,
        }

    def _calculate_group_ratio(self, reference_group: str, other_group: str) -> dict[str, float | str]:
        """
        Calculate and return the DAI ratios between a reference group and another group.
        Ensures both curves are evaluated over the same x-range.

        Args:
            reference_group: The name of the reference group.
            other_group: The name of the other group to compare against the reference group.

        Returns:
            A dictionary containing the full and decision DAI ratios.
        """
        if reference_group not in self.group_curves or other_group not in self.group_curves:
            raise ValueError("Group curves not found. Please run evaluation first.")

        ref_curve = self.group_curves[reference_group]
        oth_curve = self.group_curves[other_group]

        x_ref, y_ref = ref_curve["x"], ref_curve["y"]
        x_oth, y_oth = oth_curve["x"], oth_curve["y"]

        # Determine common x-range
        x_min = max(x_ref.min(), x_oth.min())
        x_max = min(x_ref.max(), x_oth.max())

        # Trim both curves to common range
        ref_mask = (x_ref >= x_min) & (x_ref <= x_max)
        oth_mask = (x_oth >= x_min) & (x_oth <= x_max)

        ref_trimmed = np.column_stack([x_ref[ref_mask], y_ref[ref_mask]])
        oth_trimmed = np.column_stack([x_oth[oth_mask], y_oth[oth_mask]])

        # Compute areas and ratios
        a1, da1 = self._area_under_curve(ref_trimmed)
        a2, da2 = self._area_under_curve(oth_trimmed)

        ratio = (a2 - a1) / a1 if a1 != 0 else "N/A"
        decision_ratio = (da2 - da1) / da1 if da1 != 0 else "N/A"

        return {
            "Full": ratio,
            "Decision": decision_ratio,
            "Reference Area": a1,
            "Comparison Area": a2,
            "Reference Decision Area": da1,
            "Comparison Decision Area": da2,
        }

    def _calculate_group_ratios(self, groups: list[str], rerun: bool = True) -> None:
        for group1, group2 in itertools.permutations(groups, 2):
            key = (group1, group2)
            if rerun or key not in self.group_ratios:
                self.group_ratios[key] = self._calculate_group_ratio(group1, group2)

    def _check_group(self, group: str, reference: bool = False) -> None:
        if group not in self.groups.keys():
            raise KeyError(
                f"Invalid group name provided{' for reference_group' if reference else ''}."
                f" Valid group names are {list(self.groups.keys())}"
            )

    def _validate_group_inputs(self, groups: list[str] | str | None) -> list[str]:
        if groups is not None:
            if isinstance(groups, str):
                groups = [groups]
            for group in groups:
                self._check_group(group)
        else:
            groups = list(self.groups.keys())
        return groups

    def _validate_reference_group_inputs(self, reference_group: str, groups: list[str] | str | None) -> list[str]:
        self._check_group(reference_group, reference=True)
        return list(set(self._validate_group_inputs(groups)) | {reference_group})

    def _evaluate_groups(
        self,
        models: list[ProbabilisticModel] | ProbabilisticModel | None,
        feature_list: list[str] | None,
        predictions_col: str | None,
        reference_group: str,
        groups: None | str | list[str] = None,
        rerun: bool = True,
        n_jobs: int = -1,
        using_models: bool = True,
    ) -> None:
        self.n_jobs = n_jobs
        groups = self._validate_reference_group_inputs(reference_group, groups)
        pbar = tqdm(groups, desc="Evaluating", total=len(groups), position=0)
        for group in pbar:
            pbar.set_description(f"Evaluating {group} group")
            if (rerun or group not in self.group_scores.keys()) and using_models:
                self.group_scores[group] = self._get_scores_from_models(group, models, feature_list)
            elif (rerun or group not in self.group_scores.keys()) and not using_models:
                self.group_scores[group] = self.groups[group](self.cohort)[predictions_col].to_numpy()
            if rerun or group not in self.group_ksteps.keys():
                self.group_ksteps[group] = self._get_group_ksteps(group)
            if rerun or group not in self.group_curves.keys():
                self.group_curves[group] = self._calculate_group_curve(group)
        self._calculate_group_ratios(groups, rerun)

    def evaluate_groups_by_predictions(
        self,
        predictions_col: str,
        reference_group: str,
        groups: list[str] | str | None = None,
        rerun: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """
        Calculate the DAI for group(s) based on pre-calculated model predictions

        Args:
            predictions_col: The column name in the cohort DataFrame that contains the model predictions
            reference_group: The reference group name
            groups: Single group name or list of group names to compare. If None, all groups will be evaluated
            rerun: Whether to rerun the evaluation even if results already exist for any of the groups
            n_jobs: The number of jobs to run in parallel
        """
        self._evaluate_groups(
            models=None,
            feature_list=None,
            predictions_col=predictions_col,
            reference_group=reference_group,
            groups=groups,
            rerun=rerun,
            n_jobs=n_jobs,
            using_models=False,
        )

    def evaluate_groups_by_models(
        self,
        models: list[ProbabilisticModel] | ProbabilisticModel,
        feature_list: list[str],
        reference_group: str,
        groups: list[str] | str | None = None,
        rerun: bool = True,
        n_jobs: int = -1,
    ) -> None:
        """
        Calculate the DAI for group(s) based on model objects

        Args:
            models: A single model object or a list of trained model objects that have a `predict_proba` method
            feature_list: A list of column names in the cohort DataFrame to be used as features for prediction
            reference_group: The reference group name
            groups: Single group name or list of group names to compare. If None, all groups will be evaluated
            rerun: Whether to rerun the evaluation even if results already exist for the groups
            n_jobs: The number of jobs to run in parallel
        """
        if not isinstance(models, list):
            models = [models]
        self._evaluate_groups(
            models=models,
            feature_list=feature_list,
            predictions_col=None,
            reference_group=reference_group,
            groups=groups,
            rerun=rerun,
            n_jobs=n_jobs,
            using_models=True,
        )

    def get_plot(
        self,
        reference_group: str,
        groups: list[str] | str | None = None,
        style: str = "darkgrid",
        **kwargs: dict,
    ) -> plt.Figure:
        """
        Plot DA curves for selected groups with one reference group highlighted.
        Uses a seaborn theme for clean defaults. Ratios and thresholds are added
        to legend labels.

        Args:
            reference_group (str): Name of the reference group.
            groups (list[str] | str | None): A single group name, list of names, or None (all other groups).
            style (str): Seaborn theme (e.g. 'whitegrid', 'darkgrid', 'ticks', etc.)
            **kwargs: Additional keyword arguments passed to `sns.set_theme()`

        Returns:
            (plt.Figure): The final figure object.
        """
        sns.set_theme(style=style, **kwargs)

        # Validate inputs
        all_groups = self._validate_reference_group_inputs(reference_group, groups)

        # Ensure curves exist
        for group in all_groups:
            if group not in self.group_curves:
                raise ValueError(f"Curve data for group '{group}' not found. Please run evaluation first.")

        # Sort groups to ensure reference group is first in the legend
        all_groups = sorted(all_groups, key=lambda g: 0 if g == reference_group else 1)

        # Prepare figure
        fig, ax = plt.subplots(figsize=(7, 5))

        # Determine limits
        x_max = max(np.max(self.group_curves[g]["x"]) for g in all_groups)
        y_max = max(np.max(self.group_curves[g]["y"]) for g in all_groups)
        ax.set_xlim(0, x_max * 1.05)
        ax.set_ylim(0, y_max * 1.05)

        # Plot each group
        for group in all_groups:
            curve = self.group_curves[group]
            is_ref = group == reference_group

            # Lookup ratio and threshold
            ratio_label = ""
            key = (reference_group, group)
            if not is_ref and hasattr(self, "group_ratios") and key in self.group_ratios:
                full_ratio = self.group_ratios[key].get("Full")
                if full_ratio != "N/A" and full_ratio is not None:
                    ratio_label = f" | Ratio: {full_ratio:+.3f}"
                decision_ratio = self.group_ratios[key].get("Decision")
                if decision_ratio != "N/A" and decision_ratio is not None:
                    ratio_label += f" ({decision_ratio:+.3f} decision)"

            det_threshold = self.groups[group].det_threshold or self.det_feature.threshold
            threshold_label = f" | Thr: {det_threshold:.3f}"

            # Construct legend label
            label = f"{group}{' (Reference Group)' if is_ref else ''}{threshold_label}{ratio_label}"

            # Plot with style distinction
            width = 2.0 if is_ref else 1.2
            ax.plot(curve["x"], curve["y"], "-", lw=width, label=label)
            ax.scatter(curve["x"], curve["y"], s=10)

        # Labels
        ax.set_xlabel(self.model_name)
        ax.set_ylabel("Proportion above deterioration threshold")

        # Decision boundary marker
        ax.axvline(self.decision_boundary, ls="--", lw=0.8, color="blue", alpha=0.6)
        ax.axvspan(self.decision_boundary, ax.get_xlim()[1], facecolor="blue", alpha=0.05)

        # Legend along bottom, ref group leftmost
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            fontsize=8,
            ncol=2,
            frameon=False,
        )

        # Title includes threshold column
        ax.set_title(f"DA Curves ({self.model_name}) | Feature: {self.det_feature.label}", fontsize=12, pad=10)

        fig.tight_layout(rect=[0, 0.05, 1, 1])  # Add extra room for legend
        plt.close(fig)
        return fig

    def get_ratios(self) -> pd.DataFrame:
        """
        Get the DataFrame of all group ratios.

        Returns:
            (pd.DataFrame): A DataFrame containing the ratios between all pairs of groups.
        """
        df = pd.DataFrame(self.group_ratios).T
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Reference", "Comparison"])
        return df

    def get_curves(
        self, groups: list[str] | str | None = None
    ) -> dict[str, float | np.ndarray] | dict[str, dict[str, float | np.ndarray]]:
        """
        Get the curve data for the (specified) group(s).

        Args:
            groups (list[str] | str | None): The name(s) of the group(s) to get curve data for, or None for all groups.

        Returns:
            (dict[str, CurveData]): A dictionary containing the curve data for each specified group.
        """
        groups = self._validate_group_inputs(groups)
        for group in groups:
            if group not in self.group_curves:
                raise ValueError(f"Curve data for group '{group}' not found. Please run evaluation first.")
        if len(groups) == 1:
            return self.group_curves[groups[0]]
        else:
            return {group: self.group_curves[group] for group in groups}

    def get_bin_samples(self, groups: list[str] | str | None = None) -> dict[float, int] | dict[str, dict[float, int]]:
        """
        Returns the number of samples used for each bin for the (specified) group(s).

        Args:
            groups (list[str] | str | None): The name(s) of the group(s) to get bin sample information for, or None for
            all groups.

        Returns:
            (dict[float, int] | dict[str, dict[float, int]]): A dictionary mapping bin values to the number of samples
            used.
        """
        groups = self._validate_group_inputs(groups)
        for group in groups:
            if group not in self.group_bin_samples:
                raise ValueError(f"No bin information available for group '{group}'. Run evaluation first.")
        if len(groups) == 1:
            return self.group_bin_samples[groups[0]]
        else:
            return {group: self.group_bin_samples[group] for group in groups}

    def get_sub_optimal_bins(
        self, groups: list[str] | str | None = None
    ) -> dict[float, int] | dict[str, dict[float, int]]:
        """
        Returns information about sub-optimal bins for the (specified) group(s).

        Args:
            groups (list[str] | str | None): The name(s) of the group(s) to get sub-optimal bin information for, or None
            for all groups.

        Returns:
            (dict[float, int] | dict[str, dict[float, int]]): A dictionary mapping bin scores to the number of samples
            used for sub-optimal bins.
        """
        groups = self._validate_group_inputs(groups)
        for group in groups:
            if group not in self.group_sub_optimal_bins:
                raise ValueError(f"No bin information available for group '{group}'. Run evaluation first.")
        if len(groups) == 1:
            return self.group_sub_optimal_bins[groups[0]]
        else:
            return {group: self.group_sub_optimal_bins[group] for group in groups}

    def get_failed_bins(self, groups: list[str] | str | None = None) -> dict[float, int] | dict[str, dict[float, int]]:
        """
        Returns information about failed bins for the (specified) group(s).

        Args:
            groups (list[str] | str | None): The name(s) of the group(s) to get failed bin information for, or None for
            all groups.

        Returns:
            (dict[float, int] | dict[str, dict[float, int]]): A dictionary mapping bin scores to the number of samples
            used for failed bins.
        """
        groups = self._validate_group_inputs(groups)
        for group in groups:
            if group not in self.group_failed_bins:
                raise ValueError(f"No bin information available for group '{group}'. Run evaluation first.")
        if len(groups) == 1:
            return self.group_failed_bins[groups[0]]
        else:
            return {group: self.group_failed_bins[group] for group in groups}
