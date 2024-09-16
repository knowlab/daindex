import numpy as np
import pytest
from sklearn.neighbors import KernelDensity

import daindex
from daindex.core import kde_estimate  # Adjust the import based on your actual module structure


def test_kde_estimate_default():
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    kde, bandwidth = kde_estimate(X, optimise_bandwidth=False)
    assert isinstance(kde, KernelDensity)
    assert bandwidth == 1.0


def test_kde_estimate_custom_bandwidth():
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    kde, bandwidth = kde_estimate(X, bandwidth=0.5, optimise_bandwidth=False)
    assert isinstance(kde, KernelDensity)
    assert bandwidth == 0.5


def test_kde_estimate_scott_bandwidth():
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    kde, bandwidth = kde_estimate(X, bandwidth="scott", optimise_bandwidth=False)
    assert isinstance(kde, KernelDensity)
    assert bandwidth == "scott"


def test_kde_estimate_silverman_bandwidth():
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    kde, bandwidth = kde_estimate(X, bandwidth="silverman", optimise_bandwidth=False)
    assert isinstance(kde, KernelDensity)
    assert bandwidth == "silverman"


def test_kde_estimate_optimise_bandwidth(mocker):
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    mocker.patch("daindex.core.gridsearch_bandwidth", return_value=0.3)
    kde, bandwidth = kde_estimate(X, optimise_bandwidth=True)
    assert isinstance(kde, KernelDensity)
    assert bandwidth == 0.3


def test_kde_estimate_different_kernel():
    X = np.random.normal(loc=0, scale=1, size=(100, 1))
    kde, bandwidth = kde_estimate(X, kernel="linear")
    assert isinstance(kde, KernelDensity)
    assert kde.kernel == "linear"


if __name__ == "__main__":
    pytest.main()
