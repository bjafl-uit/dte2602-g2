"""Module for measuring model performance.

This module contains functions for measuring the performance of machine
learning models. The functions in this module calculate metrics such as
accuracy, precision, recall, and confusion matrices.

Methods
-------
mean_accuracy
    Calculate the mean accuracy of a prediction.
weighted_mean_accuracy
    Calculate the frequency weighted mean accuracy of a prediction.
confusion_matrix
    Generate the confusion matrix.
precision_recall
    Calculate precision (specificity) and recall (sensitivity).
prediction_score
    Calculate the prediction score.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Any, TYPE_CHECKING, Optional, Literal, Union, Iterable
import warnings
import matplotlib.pyplot as plt
import multiprocessing as mp

if TYPE_CHECKING:
    from . import DataSplitter

from models import MLModel
from .plot import plot_confusion_matrix


def mean_accuracy(
        y_true: NDArray,
        y_pred: NDArray
) -> float:
    """Calculate the mean accuracy of a prediction.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    accuracy : float or np.ndarray
        The mean accuracy of the prediction.

    Notes
    -----
    - See function precision_recall for more advanced accuracy metrics.
    """
    return np.mean(y_true == y_pred)


def weighted_mean_accuracy(
        y_true: NDArray,
        y_pred: NDArray,
) -> float:
    """Calculate the frequency weighted mean accuracy of a prediction.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.

    Returns
    -------
    accuracy : float
        The weighted mean accuracy of the prediction, where the weights are
        the frequency of each class label.
    """
    labels = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    freq = np.array([np.sum(y_true == lbl) for lbl in labels])
    acc = np.array([
        np.mean(y_true[y_pred == lbl] == lbl) if lbl in labels_pred else 0
        for lbl in labels
    ])
    w_mean = np.sum(freq * acc) / np.sum(freq)
    return w_mean


def confusion_matrix(
        y_true: NDArray,
        y_pred: NDArray,
        labels: Optional[list] = None,
        return_labels: bool = False
) -> NDArray:
    """Generate the confusion matrix.    
    
    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    labels : list, optional
        The class labels.
    return_labels : bool, default=False
        Whether to return a vector of the sorted labels.

    Returns
    -------
    cm : np.ndarray
        The confusion matrix. The rows correspond to the true labels, and the
        columns correspond to the predicted labels. The labels are sorted in
        ascending order. If labels is None, the unique labels in y_true and
        y_pred are used.
    labels : np.ndarray
        The sorted class labels. Only returned if return_labels is True.

    Notes
    -----
    - The confusion matrix is a square matrix, where the rows correspond to the
        true labels, and the columns correspond to the predicted labels.
    - If labels is None, the unique labels in y_true and y_pred are used.
    - The diagonal of the confusion matrix contains the number of true positives
        for each class, while the off-diagonal elements contain the number of
        false positives.
    """
    labels = labels if labels is not None else \
        np.unique(np.concatenate([y_true, y_pred]))
    labels = np.sort(labels)
    y_true_int = np.argmax(y_true.reshape(-1, 1) == labels, axis=1)
    y_pred_int = np.argmax(y_pred.reshape(-1, 1) == labels, axis=1)
    n_lbls = len(labels)
    cm = np.zeros((n_lbls, n_lbls), dtype=int)
    for r in range(n_lbls):
        for c in range(n_lbls):
            cm[r, c] = np.sum((y_true_int == r) & (y_pred_int == c))

    if return_labels:
        return cm, labels
    return cm


def precision_recall(
        y_true: NDArray,
        y_pred: NDArray,
        labels: Optional[list] = None,
        average: Literal['micro', 'macro', None] = 'macro',
        return_vals: Literal['precision', 'recall', 'both'] = 'precision'
) -> float | tuple[float, float]:
    """Calculate precision (specificity) and recall (sensitivity).

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    labels : list, optional
        The class labels.
    average : str, default='macro'
        The type of average to calculate. 'micro' calculates the precision and
        recall for the whole dataset, while 'macro' calculates the precision and
        recall for each class and averages the results.
    return_vals : str, default='precision'
        The values to return, either 'precision', 'recall', or 'both'.


    Returns
    -------
    precision : float | NDArray
        The precision value. Returned as an array of percicion values for all
        classes if average is None, else as an average float (see Notes).
        Percision is returned alone if return_vals is 'precision', or as the
        first item in a tuple if return_vals is 'both'.
    recall : float | NDArray
        The recall value. Returned as an array of recall values for all
        classes if average is None, else as an average float (see Notes).
        Recall is returned alone if return_vals is 'recall', or as the second
        item in a tuple if return_vals is 'both'.

    Warnings
    --------
    UserWarning
        If a class has no predicted or true samples, the precision or recall
        value for that class is set to 0 (to avoid division by 0).
        
    Notes
    -----
    - This function utilizes the confusion_matrix function to calculate the
        precision and recall values. Note that labels are sorted in ascending
        order, and the precision and recall values are returned in the same
        order when average is None. See Notes in the confusion_matrix function.
    - The precision is calculated as the number of true positives divided by
        the sum of true positives and false positives.
    - The recall is calculated as the number of true positives divided by the
        sum of true positives and false negatives.
    - The 'micro' average calculates the precision and recall for the whole
        dataset.
    - The 'macro' average calculates the precision and recall for each class
        and averages the results.
    - If average is None, the precision and recall values are returned for
        each class, in ascending order.
    """
    cm = confusion_matrix(y_true, y_pred, labels)

    def calc_precision_recall(axis):
        sum_axis = np.sum(cm, axis=axis)
        if (zero_mask := sum_axis == 0).any():
            sum_axis = sum_axis.astype(float)
            sum_axis[zero_mask] = np.inf
            warnings.warn(
                f"Class(es) {np.where(zero_mask)[0]} have no " +
                f"{'predicted' if axis == 0 else 'true'} samples. " +
                f"{'Precision' if axis == 0 else 'Recall'} " +
                "for these set to 0 (avoid div 0)."
            )
        return np.diag(cm) / sum_axis

    if average == 'micro':
        return np.sum(np.diag(cm)) / np.sum(cm)

    if return_vals in ['precision', 'both']:
        precision = calc_precision_recall(0)
    if return_vals in ['recall', 'both']:
        recall = calc_precision_recall(1)

    if average == 'macro':
        if return_vals == 'both':
            return precision.mean(), recall.mean()
        return precision.mean() if return_vals == 'precision' \
            else recall.mean()

    if return_vals == 'both':
        return precision, recall
    return precision if return_vals == 'precision' else recall


def prediction_score(
        y_true: NDArray,
        y_pred: NDArray,
        score: Literal['b-score', 'f-score'] = 'b-score',
        mean: bool = True
) -> float:
    """Calculate the prediction score.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    score : str
        The score to calculate. 'b-score' calculates the balanced score, while
        'f-score' calculates the frequency score.

    Returns
    -------
    score : float
        The prediction score, either b-score or f-score.

    Notes
    -----
    - The balanced score is the mean of the precision and recall values.
    - The frequency score is the harmonic mean of the precision and recall
        values.
    - The precision and recall values are calculated using the precision_recall
        function.
    """
    avg = 'macro' if mean else None
    p, r = precision_recall(y_true, y_pred, average=avg, return_vals='both')
    p_r_sum = p + r

    if score == 'b-score':
        return (p + r) / 2
    
    if score == 'f-score':
        mask0 = p_r_sum == 0
        f_score = np.zeros_like(p)
        f_score[~mask0] = 2 * (p[~mask0]*r[~mask0]) / p_r_sum[~mask0]
        return f_score

