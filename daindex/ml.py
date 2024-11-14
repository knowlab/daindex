import warnings
from typing import Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from daindex.core import deterioration_index


@runtime_checkable
class ProbabilisticModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def obtain_da_index(
    df,
    cohort_name,
    scores,
    score_bounds: tuple[float, float],
    det_feature: str,
    det_threshold: float,
    det_label: str,
    det_feature_func: Callable,
    det_list_lengths: list[int],
    min_det_v: float,
    max_det_v: float,
    is_discrete: bool,
    reverse: bool,
    optimise_bandwidth: bool,
) -> tuple[int, float]:
    """
    Calculates the Deterioration Allocation Index (DAI) for a given cohort.

    Args:
        df: The DataFrame containing the data.
        cohort_name: The name of the cohort.
        scores: The list of scores corresponding to the DataFrame rows.
        score_bounds: The lower and upper bounds for the scores to be considered.
        det_feature: The feature to be used for DAI calculation.
        det_threshold: The threshold value for the deterioration index.
        det_label: The label for the deterioration index.
        det_feature_func: A function to apply to each row to extract the feature value.
        det_list_lengths: A list of acceptable lengths for the det_list, in descending order.
        min_det_v: The minimum value for the deterioration index.
        max_det_v: The maximum value for the deterioration index.
        is_discrete: Whether the feature is discrete.
        reverse: Whether to reverse the order of the feature values.
        optimise_bandwidth: Whether to search for the optimal bandwidth.

    Returns:
        A tuple containing:
            - int: The length of the det_list.
            - float: The k-step value from the deterioration index calculation.

    Raises:
        UserWarning: If the number of samples is sub-optimal or insufficient for DAI calculation.

    """
    # sort det_list_lengths into descending order
    det_list_lengths.sort(reverse=True)
    lb, ub = score_bounds

    det_list = []
    i = 0
    for idx, r in df.iterrows():
        p = scores[i]
        if lb <= p <= ub:
            if det_feature_func is not None:
                det_list.append(det_feature_func(r))
            else:
                det_list.append(r[det_feature])
        i += 1
    for det_list_length in det_list_lengths:
        if len(det_list) >= det_list_length:
            if det_list_length != det_list_lengths[0]:
                warnings.warn(
                    f"Sub-optimal number of samples for DAI calculation, {len(det_list)} is acceptable but {det_list_lengths[0]} is preferred."
                )
            break
        else:
            warnings.warn(f"Insufficient number of samples for DAI calculation, {len(det_list)} < {det_list_length}.")
            return len(det_list), 0

    X = np.array(det_list)
    di_ret = deterioration_index(
        X[~np.isnan(X)].reshape(-1, 1),
        min_det_v,
        max_det_v,
        threshold=det_threshold,
        plot_title=f"{cohort_name} | {det_label}",
        is_discrete=is_discrete,
        reverse=reverse,
        optimise_bandwidth=optimise_bandwidth,
        do_plot=False,
    )
    return len(det_list), di_ret["k-step"]


def get_scores(models: list[ProbabilisticModel], df: pd.DataFrame, feature_list: list[str]) -> np.ndarray:
    """
    Computes the mean predicted probabilities for a list of models.

    Args:
        models: A list of trained model objects that have a `predict_proba` method.
        df: A DataFrame containing the data to be used for predictions.
        feature_list: A list of column names in `df` to be used as features for prediction.

    Returns:
        An array of mean predicted probabilities for the positive class.
    """
    predicted_probs = np.array([m.predict_proba(df[feature_list].to_numpy()) for m in models])
    return predicted_probs[:, :, 1].mean(axis=0)


def get_da_values_on_models(
    df: pd.DataFrame,
    cohort_name: str,
    model: list[ProbabilisticModel] | ProbabilisticModel,
    feature_list: list[str],
    det_feature: str,
    det_threshold: float,
    det_label: str = None,
    det_feature_func: Callable = None,
    det_list_lengths: list[int] = [20, 10, 5],
    steps: int = 50,
    is_discrete: bool = False,
    reverse: bool = False,
    optimise_bandwidth: bool = True,
    score_margin_multiplier: float = 5.0,
) -> list[tuple[int, float]]:
    """
    Compute the deterioration index on the predictions of the model.

    Args:
        df: The DataFrame containing the data.
        cohort_name: The name of the cohort being analyzed.
        model: The predictive model or list of models to be used.
        feature_list: List of feature names to be used for prediction.
        det_feature: The feature name representing the deterioration metric.
        det_threshold: The threshold value for determining deterioration.
        det_label: The label for the deterioration index. Defaults to None to be replaced by `det_feature`.
        det_feature_func: The function to be used to extract the deterioration feature. Defaults to None as we assume it is in the `df`.
        det_list_lengths: The list of acceptable lengths to be used for the deterioration index calculation. The first element is the preferred minimum length. Defaults to [20, 10, 5].
        steps: The number of steps for the deterioration index calculation. Defaults to 50.
        is_discrete: Flag indicating if the deterioration feature is discrete. Defaults to False.
        reverse: Flag indicating if the deterioration index should be reversed. Defaults to False.
        optimise_bandwidth: Flag indicating if the bandwidth of the kde should be searched. Defaults to True.
        score_margin_multiplier: The multiplier to be used for the score margin, 1.0 results in no overlap, some overlap works well, essentially this smooths. Defaults to 5.0.

    Returns:
        A list of deterioration index values computed at each step,
        the length of the list of scores within the score range (there is a small upper and lower bound applied)
        and finally the da index value itself.
    """

    det_label = det_label or det_feature
    models = model if isinstance(model, list) else [model]
    ret = []

    min_det_v = np.min(df[det_feature])
    max_det_v = np.max(df[det_feature])

    scores = get_scores(models, df, feature_list)
    step_scores = np.linspace(0, 1, steps + 1)[1:] - 1 / (2 * steps)

    for s in step_scores:
        step_score_bounds = (
            max(0.0, s - (1 / (2 * steps)) * score_margin_multiplier),
            min(1.0, s + (1 / (2 * steps)) * score_margin_multiplier),
        )
        ret.append(
            (
                s,
                *obtain_da_index(
                    df,
                    cohort_name,
                    scores,
                    step_score_bounds,
                    det_feature,
                    det_threshold,
                    det_label,
                    det_feature_func,
                    det_list_lengths,
                    min_det_v,
                    max_det_v,
                    is_discrete,
                    reverse,
                    optimise_bandwidth,
                ),
            )
        )

    return ret


def get_da_values_on_predictions(
    df: pd.DataFrame,
    cohort_name: str,
    preds_col: str,
    det_feature: str,
    det_threshold: float,
    det_label: str = None,
    det_feature_func: Callable = None,
    det_list_lengths: list[int] = [20, 10, 5],
    steps: int = 50,
    is_discrete: bool = False,
    reverse: bool = False,
    optimise_bandwidth: bool = True,
    score_margin_multiplier: float = 5.0,
) -> list[tuple[int, float]]:
    """
    Compute the deterioration index on the predictions of the model.

    Args:
        df: The DataFrame containing the data.
        cohort_name: The name of the cohort being analyzed.
        preds_col: The column name containing the predictions.
        det_feature: The feature name representing the deterioration metric.
        det_threshold: The threshold value for determining deterioration.
        det_label: The label for the deterioration index. Defaults to None to be replaced by `det_feature`.
        det_feature_func: The function to be used to extract the deterioration feature. Defaults to None as we assume it is in the `df`.
        det_list_lengths: The list of acceptable lengths to be used for the deterioration index calculation. The first element is the preferred minimum length. Defaults to [20, 10, 5].
        steps: The number of steps for the deterioration index calculation. Defaults to 50.
        is_discrete: Flag indicating if the deterioration feature is discrete. Defaults to False.
        reverse: Flag indicating if the deterioration index should be reversed. Defaults to False.
        optimise_bandwidth: Flag indicating if the bandwidth of the kde should be searched. Defaults to True.
        score_margin_multiplier: The multiplier to be used for the score margin, 1.0 results in no overlap, some overlap works well, essentially this smooths. Defaults to 1.5.

    Returns:
        A list of deterioration index values computed at each step,
        the length of the list of scores within the score range (there is a small upper and lower bound applied)
        and finally the da index value itself.
    """

    det_label = det_label or det_feature
    ret = []

    min_det_v = np.min(df[det_feature])
    max_det_v = np.max(df[det_feature])

    scores = df[preds_col].values
    step_scores = np.linspace(0, 1, steps + 1)[1:] - 1 / (2 * steps)

    for s in step_scores:
        step_score_bounds = (
            max(0.0, s - (1 / (2 * steps)) * score_margin_multiplier),
            min(1.0, s + (1 / (2 * steps)) * score_margin_multiplier),
        )
        ret.append(
            (
                s,
                *obtain_da_index(
                    df,
                    cohort_name,
                    scores,
                    step_score_bounds,
                    det_feature,
                    det_threshold,
                    det_label,
                    det_feature_func,
                    det_list_lengths,
                    min_det_v,
                    max_det_v,
                    is_discrete,
                    reverse,
                    optimise_bandwidth,
                ),
            )
        )

    return ret
