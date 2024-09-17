import logging
from typing import Any, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KernelDensity


def vis_plot(
    X: np.ndarray,
    x_d: np.ndarray,
    logprobs: np.ndarray,
    ylabel: str = "Percentage/PDF",
    title: str = "",
    vlines: list[float] = [],
    vline_colors: list[str] = [],
    vline_labels: list[str] = [],
    hist: bool = True,
    reverse: bool = False,
) -> None:
    """
    plot probability density function (PDF) and histogram for comparison
    """
    if hist:
        pd.DataFrame(X).rename(columns={0: "Histogram"}).plot.hist(
            bins=20, alpha=0.5, color="goldenrod", **{"density": True}, figsize=(18, 16)
        )
    else:
        plt.figure(figsize=(8, 2), dpi=100)
    plt.fill_between(x_d, np.exp(logprobs), alpha=0.5, label="Probability Density Function")
    # plt.plot(x, np.full_like(x, -0.005), '|k', markeredgewidth=1)
    for vl, c, label in zip(vlines, vline_colors, vline_labels):
        plt.axvline(x=vl, color=c, linestyle="--", label=f"{label}:{vl}")

    if len(vlines) > 0:
        if not reverse:
            plt.axvspan(vlines[-1], x_d[-1], facecolor="b", alpha=0.05)
        else:
            plt.axvspan(x_d[0], vlines[-1], facecolor="b", alpha=0.05)

    plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.yticks([])
    # plt.ylim((0, .03))
    # plt.xlim((10, 40))


def gridsearch_bandwidth(X: np.ndarray) -> float:
    """
    Search for the best bandwith for the KDE
    """
    bandwidths = np.linspace(0, 1, 20)
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), {"bandwidth": bandwidths}, cv=KFold(5))
    grid.fit(X)
    return grid.best_params_["bandwidth"]


def get_threshold_index(
    threshold: float, low_bound: float, is_discrete: bool, prev_val_offset: float, step: float, boundary_offset: float
) -> int:
    """
    Calculate threshold value index
    """
    threshold_index = int((threshold - low_bound) / step)
    if threshold == low_bound + boundary_offset:
        logging.info(f"adjust boundary {threshold}")
        threshold_index = int((threshold - low_bound - boundary_offset) / step)
    elif is_discrete:
        # discrete values will lead to PDF shape like pulses,
        # it's important to start with the valley between the pulse you want to include and the one before that
        # the following is to find the valley index
        threshold_index_prev = int((threshold - prev_val_offset - low_bound) / step)
        threshold_index = int(threshold_index_prev + (threshold_index - threshold_index_prev) / 2)
    return threshold_index


def search_for_zero_mass_index(kde: KernelDensity, min_v: float, n_samples: int = 100) -> tuple[int, float]:
    """
    Search near zero probability mass for boundary adjustment
    """
    bins = np.linspace(min_v - 10, min_v, n_samples)
    pd_vals = np.exp(kde.score_samples(bins.reshape(-1, 1)))
    first_zero_idx = np.max(np.where(pd_vals < 1e-10))
    return first_zero_idx, bins[first_zero_idx]


def kde_estimate(
    X: np.ndarray,
    bandwidth: float | Literal["scott", "silverman"] = 1.0,
    kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
    optimise_bandwidth: bool = True,
) -> tuple[KernelDensity, float | Literal["scott", "silverman"]]:
    """
    Kernel density estimation to get probability
    """
    if optimise_bandwidth:
        bandwidth = gridsearch_bandwidth(X)
        logging.info(f"learned best bandwidth {bandwidth}")

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
    kde.fit(X)

    return kde, bandwidth


def deterioration_index(
    X: np.ndarray,
    low_bound: float,
    up_bound: float,
    threshold: float,
    n_samples: int = 10000,
    plot_title: str = "",
    is_discrete: bool = False,
    prev_discrete_value_offset=1,
    weight_sum_steps=10,
    reverse: bool = False,
    bandwidth: float | Literal["scott", "silverman"] = 1.0,
    kernel: Literal["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"] = "gaussian",
    optimise_bandwidth: bool = True,
    do_plot: bool = True,
) -> dict[str, int]:
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
    kde, fitted_bandwith = kde_estimate(X, bandwidth=bandwidth, kernel=kernel, optimise_bandwidth=optimise_bandwidth)

    # detect pulse like PDF
    if fitted_bandwith < 0.1:
        # force is_discrete because the bestfitted_bandwith_bw would lead to one
        logging.info(f"FORCED to be discrete because the bandwidth is {fitted_bandwith}")
        is_discrete = True
    elif fitted_bandwith > 0.7:
        # force is_discrete to be false because the fitted_bandwith would lead to one
        logging.info(f"FORCED to be NOT discrete because the bandwidth is {fitted_bandwith}")
        is_discrete = False

    # automatically adjust on boundaries
    _, adjusted_min = search_for_zero_mass_index(kde, low_bound)
    logging.info(f"adjusted min val {adjusted_min}")
    boundary_offset = low_bound - adjusted_min
    # orig_low_bound = low_bound
    low_bound -= boundary_offset
    up_bound += boundary_offset

    # use learned KDE estimator to get probability
    bins = np.linspace(low_bound, up_bound, n_samples)
    kd_vals = kde.score_samples(bins.reshape(-1, 1))  # Get PDF values for each x
    step_width = (up_bound - low_bound) / n_samples  # get the step
    probs = np.exp(kd_vals) * step_width  # get the approximate prob at each point using the integral of the PDF

    if do_plot:
        tidx = get_threshold_index(
            threshold,
            low_bound,
            is_discrete,
            prev_discrete_value_offset,
            step_width,
            boundary_offset=boundary_offset,
        )
        vis_plot(
            X,
            bins,
            kd_vals,
            title=plot_title,
            vlines=[threshold, round(tidx * step_width + low_bound, 2)],
            vline_colors=["r", "b"],
            vline_labels=["Threshold", "Boudary-adjusted"],
            hist=False,
            reverse=reverse,
        )

    # severity quantification
    if reverse:
        s = low_bound
        e = min(threshold, up_bound)
    else:
        s = max(threshold, low_bound)
        e = up_bound

    # 1. binary like multimorbidity num > 3, yes or no
    sq1 = stepped_severity(
        probs,
        s,
        e,
        1,
        low_bound,
        step_width,
        is_discrete,
        boundary_offset,
        prev_val_offset=prev_discrete_value_offset,
        reverse=reverse,
    )
    # 2. stepped quantification that considers higher/lower the value, more severe the patients are
    sqs = stepped_severity(
        probs,
        s,
        e,
        weight_sum_steps,
        low_bound,
        step_width,
        is_discrete,
        boundary_offset,
        prev_val_offset=prev_discrete_value_offset,
        reverse=reverse,
    )

    return {"overall-prob": round(probs.sum(), 4), "one-step": round(sq1, 4), "k-step": round(sqs, 6), "|X|": len(X)}


def stepped_severity(
    probs: np.ndarray,
    s: float,
    e: float,
    steps: int,
    low_bound: float,
    step_width: float,
    is_discrete: bool,
    boundary_offset: float,
    prev_val_offset: float = 1.0,
    reverse: bool = False,
) -> float:
    """
    To quantify severity by considering higher values as more severe.
    This is done by weighted sum by integrating probs from a threshold - s
    """

    bins = np.linspace(s, e, steps)
    # bin_probs = [probs[get_threshold_index(t, low_bound, is_discrete, step_width):].sum()  for t in bins]
    bin_probs = []
    for i, t in enumerate(bins):
        idx1 = get_threshold_index(t, low_bound, is_discrete, prev_val_offset, step_width, boundary_offset)
        if i < len(bins) - 1:
            idx2 = get_threshold_index(
                bins[i + 1], low_bound, is_discrete, prev_val_offset, step_width, boundary_offset
            )
            bin_probs.append(probs[idx1:idx2].sum())
        else:
            (
                bin_probs.append(probs[idx1:].sum())
                if not reverse
                else bin_probs.append(
                    probs[
                        idx1 : get_threshold_index(
                            e, low_bound, is_discrete, prev_val_offset, step_width, boundary_offset
                        )
                    ].sum()
                )
            )
    s = 0

    # Weight functions
    # def weight_function(x: Any) -> Union[int, Any]:
    #     return math.log(x + 2, 2)

    def weight_function(x: Any) -> Union[int, Any]:
        return x + 1

    w = 0
    for i, p in enumerate(bin_probs):
        if not reverse:
            s += weight_function(i) * p
        else:
            s += weight_function(len(bin_probs) - i) * p
        w += weight_function(i)
    return s / w


def db_ineq(di1: dict[str, int], di2: dict[str, int]) -> float:
    """
    quantify inequality
    """
    return di1["k-step"] / di2["k-step"] - 1
