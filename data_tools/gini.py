"""Gini impurity calculation functions.

This submodule contains functions for calculating Gini impurity and
impurity reduction for decision tree algorithms.

Methods
-------
gini_impurity
    Calculate Gini impurity of a vector.
gini_impurity_reduction
    Calculate the reduction in mean impurity from a binary split.
best_split_feature_value
    Find the feature and value "split" that yields highest impurity reduction.

Notes
-----
- The Gini impurity is a measure of the entropy of a dataset. It's given by the 
    probability of misclassifying an item in the dataset if it were randomly 
    labeled. The Gini impurity is calculated as 1 - sum(p_i^2), where p_i is the
    probability of item i in the dataset. A Gini impurity of 0 indicates a dataset
    where all items have the same class label. A Gini impurity close to 1 indicates
    theres almost a zero chance of randomly labeling an item correctly.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional


def gini_impurity(y: NDArray) -> float:
    """Calculate Gini impurity of a vector.

    Parameters
    ----------
    y: NDArray, integers
        1D NumPy array with class labels

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    Notes
    -----
    - The formula of Gini impurity is sum(p_i * (1 - p_i)), where p_i is the
        probability of randomly picing an item of class i in the dataset. Here 
        we calculate it in the equivalent form 1 - sum(p_i^2), since the 
        sum of p_i for all classes i is 1. 
    """
    labels = np.unique(y)
    freq = np.array([np.sum(y == label) for label in labels])
    n = y.size
    impurity = 1.0 - np.sum((freq / n) ** 2)
    return impurity

def gini_impurity_from_freq(y_frequencies: ArrayLike) -> float:
    """Calculate Gini impurity from frequency counts.

    Parameters
    ----------
    freq: ArrayLike
        1D array with class frequencies

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    Notes
    -----
    - The formula of Gini impurity is sum(p_i * (1 - p_i)), where p_i is the
        probability of randomly picing an item of class i in the dataset. Here 
        we calculate it in the equivalent form 1 - sum(p_i^2), since the 
        sum of p_i for all classes i is 1. 
    """
    freq = np.array(y_frequencies)
    n = freq.sum()
    impurity = 1.0 - np.sum((freq / n) ** 2)
    return impurity

def gini_impurity_reduction(
        y: NDArray,
        left_mask: NDArray,
        gi_y: Optional[float] = None
) -> float:
    """Calculate the reduction in mean impurity from a binary split.

    Parameters
    ----------
    y: NDArray
        1D numpy array
    left_mask: NDArray
        1D numpy boolean array, True for "left" elements, False for "right"
    gi_y: float, optional
        Gini impurity of the original (not split) dataset.
        If not provided, it is calculated from y.

    Returns
    -------
    impurity_reduction: float
        Reduction in mean Gini impurity, scalar in range [0,0.5]
        Reduction is measured as _difference_ between Gini impurity for
        the original (not split) dataset, and the _weighted mean impurity_
        for the two split datasets ("left" and "right").

    Notes
    -----

    - The impurity reduction is measured as the difference between the Gini
      impurity of the original dataset and the weighted mean impurity of the
      two split datasets ("left" and "right").
    - The Gini impurity for y may be provided as an argument to avoid
      redundant calculations when calculating impurity for multiple splits.
    """
    gi_y = gini_impurity(y)
    y_left = y[left_mask]
    y_right = y[~left_mask]
    gi_split_wt_mean = sum(gini_impurity(y_split) * y_split.size / y.size
                           for y_split in (y_left, y_right))
    gi_impurity = gi_y - gi_split_wt_mean
    return gi_impurity


def best_split_feature_value(
        X: NDArray,
        y: NDArray,
        min_samples_leaf: int = 0  #TODO
) -> tuple[float, int, float]:
    """Find feature and value "split" that yields highest impurity reduction.

    Parameters
    ----------
    X: NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y: NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction: float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index: int
        Index of X column with best feature for split.
        None if impurity_reduction = 0.
    feature_value: float
        Value of feature in X yielding best split of y
        Dataset is split using X[:,feature_index] <= feature_value
        None if impurity_reduction = 0.
        Gini impurity of y

    Notes
    -----
    The method checks every possible combination of feature and
    existing unique feature values in the dataset.
    """
    gi_y = gini_impurity(y)
    best_gi_reduction = 0.0
    best_feat_idx = None
    best_feat_val = None

    for i in range(X.shape[1]):
        for value in np.unique(X[:, i]):
            left_mask = X[:, i] <= value
            if (n_left := np.sum(left_mask)) < min_samples_leaf \
                or X.shape[0] - n_left < min_samples_leaf:
                continue 
            gi_red = gini_impurity_reduction(y, left_mask, gi_y)
            if gi_red > best_gi_reduction:
                best_gi_reduction = gi_red
                best_feat_idx = i
                best_feat_val = value

    return best_gi_reduction, best_feat_idx, best_feat_val, gi_y
