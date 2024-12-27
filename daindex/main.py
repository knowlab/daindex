import warnings
from typing import Any, Callable, Literal, Protocol, runtime_checkable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity
from tqdm.autonotebook import tqdm


@runtime_checkable
class ProbabilisticModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


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
        reverse: For calculating p(X<threshold), i.e., the smaller the measure value the more severe a patient is.
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
    ):
        self.col = col
        self.threshold = threshold
        self.label = label or col
        self.func = func
        self.is_discrete = is_discrete
        self.prev_discrete_value_offset = prev_discrete_value_offset
        self.reverse = reverse

    def __repr__(self):
        return f"DeteriorationFeature(col='{self.col}', threshold={self.threshold}, label='{self.label}')"


class Group(object):
    """
    Class to define a sub-group within the cohort upon which to calculate the DA Index.
    A list of these should be passed into the `groups` parameter of the `DAIndex` class.

    Args:
        name: The name of the group in the desired formatting to display on plots and use as a key to extract results etc.
        definition: The value(s) in the column that define the group.
            E.g. could be a list of values, a single value, string, number, etc.
            If not provided, defaults to the name.
        col: The column name in the cohort DataFrame that contains the group definition. Can be overriden by the `group_col` parameter of the `DAIndex` class.
        det_threshold: The threshold value for the deterioration index, overriding the `threshold` attribute of the `DeteriorationFeature`.
        get_group: This is an optional argument to allow passing in of a more complex function
            that returns the group DataFrame.
            If not provided, the group DataFrame is obtained by filtering the cohort DataFrame
            by the `col` and `definition`.

    Methods:
        Call the object to return the group DataFrame by operating on the cohort DataFrame.

    Examples:
        >>> cohort = pd.DataFrame({"group_col": ["group_1", "group_2", "group_1", "group_3"]})
        >>> group = Group("Group 1", "group_col", "group_1")
        >>> group(cohort)

          group_col
        0   group_1
        2   group_1

        >>> cohort = pd.DataFrame({"sex": ["M", "F", "m", "F", "f", "female"]})
        >>> group = Group("Female", "sex", ["F", "f", "female"])
        >>> group(cohort)

               sex
        1   F
        3   F
        4   f
        5   female
    """

    def __init__(
        self, name: str, definition: Any = None, col: str = None, det_threshold=None, get_group: Callable = None
    ):
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

    def __repr__(self):
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
    We compare the DAI between two groups relative to a given model's predictions. This then gives us
    a measure of how fair the model's predictions are across the two groups.

    This class should be instantiated first with a `DetertiorationFeature` object and a list of `Group` objects.
    The `evaluate_group_pair_by_predictions` or `evaluate_group_pair_by_models` methods can then be called to
    calculate the DAI for a pair of groups. Alternatively, the `evaluate_all_groups_by_predictions` or
    `evaluate_all_groups_by_models` methods can be called to calculate the DAI for all groups relative to a
    single reference group.

    The results can be accessed using the `get_group_ratios` and `get_group_figures` methods, or printed using
    the `present_results` and `present_all_results` methods.

    Args:
        cohort: The DataFrame containing the data for the cohort.
        groups: A list of `Group` objects representing the sub-groups within the cohort.
        det_feature: A `DeteriorationFeature` object representing the feature upon which to calculate the DAI.
        group_col: Optional column name in the cohort DataFrame that contains the group definition.
            Specifying this overrides the `col` attribute of the `Group` objects.
        steps: The number of steps to use for the DAI calculation.
        score_margin_multiplier: The multiplier to use for the score margin.
        det_list_lengths: A list of acceptable lengths for the det_list, in descending order.
        bandwidth: The bandwidth to use for the KDE.
        optimise_bandwidth: Whether to search for the optimal bandwidth.
        kernel: The kernel to use for the KDE.
        n_samples: The number of samples to use for the KDE.
        weight_sum_steps: The number of bins for the weighted sum of k-step cutoffs.
        n_jobs: The number of jobs to run in parallel.
        model_name: The name of the model to use in the plots.
        decision_boundary: The decision boundary for the DA curve.

    Methods:
        setup_groups: Set up the groups for the DAI calculation.
        setup_daauc_params: Set up the parameters for the DAI calculation.
        setup_deterioration_feature: Set up the deterioration feature for the DAI calculation.
        evaluate_group_pair_by_predictions: Calculate the DAI for a pair of groups based on model predictions.
        evaluate_group_pair_by_models: Calculate the DAI for a pair of groups based on model objects.
        evaluate_all_groups_by_predictions: Calculate the DAI for all groups relative to a single reference group based on model predictions.
        evaluate_all_groups_by_models: Calculate the DAI for all groups relative to a single reference group based on model objects.
        present_results: Print the DAI results for a pair of groups.
        present_all_results: Print the DAI results for all group pairs.
        get_group_ratios: Get the DAI ratios for a pair of groups.
        get_group_figures: Get the DAI figures for a pair of groups.
        get_all_ratios: Get the DAI ratios for all group pairs.
        get_all_figures: Get the DAI figures for all group pairs.
    """

    def __init__(
        self,
        cohort: pd.DataFrame,
        groups: Group | list[Group],
        det_feature: DeteriorationFeature,
        group_col: str = None,
        steps: int = 50,
        score_margin_multiplier: float = 5.0,
        det_list_lengths: list[int] = [20, 10, 5],
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        optimise_bandwidth: bool = False,
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        n_samples: int = 10000,
        weight_sum_steps: int = 10,
        n_jobs: int = -1,
        model_name: str = None,
        decision_boundary: float = 0.5,
    ):
        self.cohort = cohort
        self.setup_groups(groups, group_col)
        self.setup_deterioration_feature(det_feature)
        self.setup_daauc_params(
            steps,
            score_margin_multiplier,
            det_list_lengths,
            bandwidth,
            optimise_bandwidth,
            kernel,
            n_samples,
            weight_sum_steps,
            n_jobs,
        )

        self.model_name = model_name or "Allocation"
        self.decision_boundary = decision_boundary

        self.group_scores = {}
        self.group_ksteps = {}
        self.group_ratios = {}
        self.group_figures = {}
        self.issues = []

    def setup_groups(self, groups: Group | list[Group], group_col: str = None) -> None:
        if not isinstance(groups, list):
            groups = [groups]
        if group_col is not None:
            groups = [Group(g.name, group_col, g.definition, g.get_group) for g in groups]
        elif any(g.col is None for g in groups):
            raise ValueError("group_col must be provided if any group objects do not have a col attribute")
        self.groups = {g.name: g for g in groups}

    def setup_daauc_params(
        self,
        steps: int = 50,
        score_margin_multiplier: float = 5.0,
        det_list_lengths: list[int] = [20, 10, 5],
        bandwidth: float | Literal["scott", "silverman"] = 1.0,
        optimise_bandwidth: bool = False,
        kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
        n_samples: int = 10000,
        weight_sum_steps: int = 10,
        n_jobs: int = -1,
    ) -> None:
        self.steps = steps
        self.score_margin_multiplier = score_margin_multiplier
        self.step_scores = np.linspace(0, 1, self.steps + 1)[1:] - 1 / (2 * self.steps)
        self.step_score_bounds = {
            s: (
                max(0.0, s - (1 / (2 * self.steps)) * self.score_margin_multiplier),
                min(1.0, s + (1 / (2 * self.steps)) * self.score_margin_multiplier),
            )
            for s in self.step_scores
        }
        det_list_lengths.sort(reverse=True)
        self.det_list_lengths = det_list_lengths
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

    def _gridsearch_bandwidth(self, X: np.ndarray) -> float:
        """
        Search for the best bandwith for the KDE
        """
        bandwidths = np.linspace(0, 1, 20)
        grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=KFold(5))
        grid.fit(X)
        return grid.best_params_["bandwidth"]

    def _kde_estimate(self, X: np.ndarray) -> KernelDensity:
        """
        Kernel density estimation to get probability
        """
        if self.optimise_bandwidth:
            bandwidth = self._gridsearch_bandwidth(X)
        else:
            bandwidth = self.bandwidth
        kde = KernelDensity(bandwidth=bandwidth, kernel=self.kernel)
        kde.fit(X)

        # detect pulse like PDF
        if bandwidth < 0.1:
            # force is_discrete because the bestfitted_bandwith_bw would lead to one
            self.is_discrete = True
        elif bandwidth > 0.7:
            # force is_discrete to be false because the fitted_bandwith would lead to one
            self.is_discrete = False

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
        elif self.is_discrete:
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

        def weight_function(x: Any) -> int | Any:
            return x + 1

        w = 0
        for i, p in enumerate(bin_probs):
            if not self.det_feature.reverse:
                s += weight_function(i) * p
            else:
                s += weight_function(len(bin_probs) - i) * p
            w += weight_function(i)
        return s / w

    def _deterioration_index(self, X: np.ndarray, group: str) -> float:
        """
        Obtain deterioration index
        X - the random sample of measurements
        low_bound/up_nbound - the boundary values of the measurement
        n_samples - number of bins to use for probability calculation. default is 2000.
        plot_title - the title of the plot, if generates plot. default is empty string
        is_discrete - whether the random sample is discrete. NB: this might be overwritten based on bandwidth learned.
            Small bandwidths will always bring out pulse like PDFs. default is False.
        prev_discrete_value_offset - the difference between the threshold and the previous legitimate value. default is 1.
        weight_sum_steps - the number of bins for weighted sum of k-step cutoffs, default is 20
        reverse - for calculating p(X<threshold), i.e., the smaller the measure value the more severe a patient is.
            default is False
        bandwidth - default bandwidth to use if not search bandwidth, default is 1
        kernel - the kernel to use for KDE, default is gaussian.
        optimise_bandwidth - whether to use grid search to find optimal bandwidth for X. default is True
        do_plot - whether to generate plots, default is True
        """

        # estimate density function
        kde = self._kde_estimate(X)

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

    def _obtain_da_index(self, group: str, score_bounds: tuple[float, float]) -> tuple[int, float]:
        """
        Calculates the Deterioration Allocation Index (DAI) for a given cohort.

        Args:
            df: The DataFrame containing the data.
            cohort_name: The name of the cohort.
            scores: The list of scores corresponding to the DataFrame rows.
            score_bounds: The lower and upper bounds for the scores to be considered.
            det_feature: The col to be used for DAI calculation.
            det_threshold: The threshold value for the deterioration index.
            det_label: The label for the deterioration index.
            det_feature_func: A function to apply to each row to extract the col value.
            det_list_lengths: A list of acceptable lengths for the det_list, in descending order.
            min_det_v: The minimum value for the deterioration index.
            max_det_v: The maximum value for the deterioration index.
            is_discrete: Whether the col is discrete.
            reverse: Whether to reverse the order of the col values.
            optimise_bandwidth: Whether to search for the optimal bandwidth.

        Returns:
            A tuple containing:
                - int: The length of the det_list.
                - float: The k-step value from the deterioration index calculation.

        Raises:
            UserWarning: If the number of samples is sub-optimal or insufficient for DAI calculation.

        """
        lb, ub = score_bounds

        det_list = []
        i = 0
        sub_opt = False

        df = self.groups[group](self.cohort)
        for idx, r in df.iterrows():
            p = self.group_scores[group][i]
            if lb <= p <= ub:
                if self.det_feature.func is not None:
                    det_list.append(self.det_feature.func(r))
                else:
                    det_list.append(r[self.det_feature.col])
            i += 1
        for det_list_length in self.det_list_lengths:
            if len(det_list) >= det_list_length:
                if det_list_length != self.det_list_lengths[0]:
                    sub_opt = True
                break
        else:
            return len(det_list), 0.0, False, True

        X = np.array(det_list)
        di_ret = self._deterioration_index(X[~np.isnan(X)].reshape(-1, 1), group)
        return len(det_list), di_ret, sub_opt, False

    def _get_group_ksteps(self, group) -> list[tuple[float, int, float]]:
        def process_step(s) -> tuple[float, int, float, bool, bool]:
            length, di_ret, sub_opt, failed = self._obtain_da_index(group, self.step_score_bounds[s])
            return (s, length, di_ret, sub_opt, failed)

        ret = Parallel(n_jobs=self.n_jobs)(
            delayed(process_step)(s)
            for s in tqdm(self.step_scores, desc=f"Calculating k-steps for '{group}' group", position=1, leave=False)
        )
        sub_opt_list = ", ".join([f"{s[0]}: {s[1]}" for s in ret if s[3]])
        failed_list = ", ".join([f"{s[0]}: {s[1]}" for s in ret if s[4]])
        if sub_opt_list or failed_list:
            message = f"\nIssues were encountered during the {group} group calculation:"
            if sub_opt_list:
                message += f"\nThere are a sub-optimal number of samples for these scores: {sub_opt_list}"
            if failed_list:
                message += f"\nThere are too few samples for these scores: {failed_list}"
            warnings.warn(message)
        return np.array([(s[0], s[1], s[2]) for s in ret if not s[4]])

    def _evaluate_group_pair(self, reference_group: str, other_group: str, rerun: bool, rerun_reference: bool) -> None:
        if rerun_reference or reference_group not in self.group_ksteps.keys():
            self.group_ksteps[reference_group] = self._get_group_ksteps(reference_group)
        if rerun or other_group not in self.group_ksteps.keys():
            self.group_ksteps[other_group] = self._get_group_ksteps(other_group)

    def _check_group_pair(self, reference_group: str, other_group: str) -> None:
        assert (
            reference_group in self.groups.keys()
        ), f"Invalid group name provided for reference_group. Valid group names are {self.groups.keys().to_list()}"
        assert (
            other_group in self.groups.keys()
        ), f"Invalid group name provided for other_group. Valid group names are {self.groups.keys().to_list()}"

    def _evaluate_group_pair_by_predictions(
        self, predictions_col: str, reference_group: str, other_group: str, rerun: bool, rerun_reference: bool
    ):
        if rerun_reference or reference_group not in self.group_scores.keys():
            self.group_scores[reference_group] = self.groups[reference_group](self.cohort)[predictions_col].to_numpy()
        if rerun or other_group not in self.group_scores.keys():
            self.group_scores[other_group] = self.groups[other_group](self.cohort)[predictions_col].to_numpy()
        self._evaluate_group_pair(reference_group, other_group, rerun, rerun_reference)

    def evaluate_group_pair_by_predictions(
        self,
        predictions_col: str,
        reference_group: str,
        other_group: str,
        rerun: bool = True,
        n_jobs: int = -1,
    ):
        self.n_jobs = n_jobs
        self._check_group_pair(reference_group, other_group)
        self._evaluate_group_pair_by_predictions(predictions_col, reference_group, other_group, rerun, rerun)
        self.group_ratios[(reference_group, other_group)], self.group_figures[(reference_group, other_group)] = (
            self.get_da_curve(reference_group, other_group)
        )
        return self.group_ratios[(reference_group, other_group)], self.group_figures[(reference_group, other_group)]

    def _get_scores(self, group: str, models: list[ProbabilisticModel], feature_list: list[str]) -> np.ndarray:
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

    def _evaluate_group_pair_by_models(
        self,
        models: list[ProbabilisticModel],
        feature_list: list[str],
        reference_group: str,
        other_group: str,
        rerun: bool,
        rerun_reference: bool,
    ):
        if rerun_reference or reference_group not in self.group_scores.keys():
            self.group_scores[reference_group] = self._get_scores(reference_group, models, feature_list)
        if rerun or other_group not in self.group_scores.keys():
            self.group_scores[other_group] = self._get_scores(other_group, models, feature_list)
        self._evaluate_group_pair(reference_group, other_group, rerun, rerun_reference)

    def evaluate_group_pair_by_models(
        self,
        models: list[ProbabilisticModel] | ProbabilisticModel,
        feature_list: list[str],
        reference_group: str,
        other_group: str,
        rerun: bool = True,
        n_jobs: int = -1,
    ):
        self.n_jobs = n_jobs
        self._check_group_pair(reference_group, other_group)
        models = models if isinstance(models, list) else [models]
        self._evaluate_group_pair_by_models(models, feature_list, reference_group, other_group, rerun, rerun)
        self.group_ratios[(reference_group, other_group)], self.group_figures[(reference_group, other_group)] = (
            self.get_da_curve(reference_group, other_group)
        )
        return self.group_ratios[(reference_group, other_group)], self.group_figures[(reference_group, other_group)]

    def _check_reference_group(self, reference_group: str) -> None:
        assert (
            reference_group in self.groups.keys()
        ), f"Invalid group name provided for reference_group. Valid group names are {self.groups.keys().to_list()}"

    def evaluate_all_groups_by_models(
        self,
        models: list[ProbabilisticModel] | ProbabilisticModel,
        feature_list: list[str],
        reference_group: str,
        rerun: bool = True,
        n_jobs: int = -1,
    ):
        self.n_jobs = n_jobs
        self._check_reference_group(reference_group)
        models = models if isinstance(models, list) else [models]
        pbar = tqdm(
            [*(self.groups.keys() - {reference_group})],
            desc="Evaluating group",
            total=len(self.groups) - 1,
            position=0,
        )
        for group in pbar:
            pbar.set_description(f"Evaluating {group} group")
            self._evaluate_group_pair_by_models(models, feature_list, reference_group, group, rerun, False)
            self.group_ratios[(reference_group, group)], self.group_figures[(reference_group, group)] = (
                self.get_da_curve(reference_group, group)
            )

    def evaluate_all_groups_by_predictions(
        self, predictions_col: str, reference_group: str, rerun: bool = True, n_jobs: int = -1
    ):
        self.n_jobs = n_jobs
        self._check_reference_group(reference_group)
        pbar = tqdm(
            [*(self.groups.keys() - {reference_group})],
            desc="Evaluating group",
            total=len(self.groups) - 1,
            position=0,
        )
        for group in pbar:
            pbar.set_description(f"Evaluating {group} group")
            self._evaluate_group_pair_by_predictions(predictions_col, reference_group, group, rerun, False)
            self.group_ratios[(reference_group, group)], self.group_figures[(reference_group, group)] = (
                self.get_da_curve(reference_group, group)
            )

    def _area_under_curve(self, w_data: np.ndarray) -> tuple[float, float]:
        """
        calculate the area under curve - do NOT do interpolation
        """
        prev = None
        area = 0.0
        decision_area = 0.0
        n_points = 0
        for r in w_data:
            if prev is not None:
                a = (r[1] + prev[1]) * (r[0] - prev[0]) / 2  # * r[2]
                area += a
                if prev[0] >= self.decision_boundary:
                    decision_area += a
                    n_points += 1
            prev = r

        if prev is not None:
            a = (r[1] + prev[1]) * (r[0] - prev[0]) / 2  # * r[2]
            area += a
            if prev[0] >= self.decision_boundary:
                decision_area += a
                n_points += 1

        return area, decision_area

    def _vis_da_indices(self, data: np.ndarray, label: str) -> tuple[float, float, np.ndarray]:
        """
        plot dot-line for approximating a DA curve
        """
        w_data = data[np.where(data[:, 1] > 0)][:, [0, 2, 1]]
        a, decision_area = self._area_under_curve(w_data)
        plt.plot(w_data[:, 0], w_data[:, 1], "-")
        plt.plot(w_data[:, 0], w_data[:, 1], "o", label=label)
        return a, decision_area, w_data

    def _calc_ratios(self, a1: float, a2: float, da1: float, da2: float) -> dict[str, float | str]:
        ratio = (a2 - a1) / a1
        decision_ratio = ((da2 - da1) / da1) if da1 != 0 else "N/A"
        return {"Full": ratio, "Decision": decision_ratio}

    def get_da_curve(
        self,
        reference_group: str,
        other_group: str,
    ) -> tuple[dict[str, float | str], plt.Figure]:
        """
        do DA curve visualisation
        """

        self._check_group_pair(reference_group, other_group)
        d1 = self.group_ksteps[reference_group]
        d2 = self.group_ksteps[other_group]

        # make two datasets even in terms of max x val
        x_min = min(np.max(d1[:, 0]), np.max(d2[:, 0]))
        d1 = np.delete(d1, np.where(d1[:, 0] > x_min), axis=0)
        d2 = np.delete(d2, np.where(d2[:, 0] > x_min), axis=0)

        # automatically set x/y limits for better viz
        # x_max = max(np.max(d1[:, 0]), np.max(d2[:, 0]))
        y_max = max(np.max(d1[:, 2]), np.max(d2[:, 2]))

        plt.xlim(0, x_min * 1.05)
        plt.ylim(0, y_max * 1.05)

        # do plots
        a1, da1, _ = self._vis_da_indices(d1, reference_group)
        a2, da2, _ = self._vis_da_indices(d2, other_group)

        ratios = self._calc_ratios(a1, a2, da1, da2)

        # figure finishing up
        plt.xlabel(self.model_name)
        det_threshold = self.groups[other_group].det_threshold or self.det_feature.threshold
        inequality_string = ">=" if not self.det_feature.reverse else "<="
        ylabel_string = f"{self.det_feature.label} {inequality_string} {det_threshold}"
        det_threshold_reference = self.groups[reference_group].det_threshold or self.det_feature.threshold
        if det_threshold != det_threshold_reference:
            ylabel_string += f" for {other_group} group,\n{self.det_feature.label} {inequality_string} {det_threshold_reference} for {reference_group} group"
        plt.ylabel(ylabel_string)

        # plot decision region
        plt.plot([self.decision_boundary, self.decision_boundary], [0, 1], "--", lw=0.8, color="g")
        plt.axvspan(self.decision_boundary, 1, facecolor="b", alpha=0.1)

        plt.legend(loc="best")

        fig = plt.gcf()
        plt.close()

        return ratios, fig

    def present_results(self, reference_group, other_group):
        ratios, fig = (
            self.group_ratios[(reference_group, other_group)],
            self.group_figures[(reference_group, other_group)],
        )
        print(f"Reference group: {reference_group}, Comparison group: {other_group}")
        print(
            f"Full AUC Ratio = {(ratios['Full'] * 100):.2f}%, Decision AUC Ratio = {(ratios['Decision'] * 100):.2f}%"
            if ratios["Decision"] != "N/A"
            else f"Full AUC Ratio = {(ratios['Full'] * 100):.2f}%, Decision AUC Ratio = N/A"
        )
        display(fig)

    def present_all_results(self):
        for group_pair in self.group_figures.keys():
            self.present_results(*group_pair)

    def get_group_ratios(self, reference_group, other_group):
        return self.group_ratios[(reference_group, other_group)]

    def get_group_figures(self, reference_group, other_group):
        return self.group_figures[(reference_group, other_group)]

    def get_all_ratios(self):
        df = pd.DataFrame(self.group_ratios).T
        df.index = pd.MultiIndex.from_tuples(df.index, names=["Reference", "Comparison"])
        return df

    def get_all_figures(self):
        return self.group_figures
