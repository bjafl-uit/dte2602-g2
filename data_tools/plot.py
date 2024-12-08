
"""Ploting functions for visualizing model performance.

This module contains functions for plotting machine learning model
properties and performance. 

Methods
-------
plot_confusion_matrix
    Plot the confusion matrix.
feature_plot
    Plot the feature distribution.
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from .color_map import DiscreteColorMap
from typing import Optional
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def plot_confusion_matrix(
        cm: NDArray,
        labels: NDArray,
        ax: Optional[plt.Axes] = None,
        show_plot: bool = True
) -> None:
    """Plot the confusion matrix.

    Parameters
    ----------
    cm : np.ndarray
        The confusion matrix. The rows should correspond to the true labels,
        and the columns correspond to the predicted labels.
    labels : np.ndarray
        The class labels.
    ax : plt.Axes, optional
        The axes to plot the confusion matrix on. If None, a new figure is
        created.
    show_plot : bool, optional
        Whether to show the plot. Default is True.
    """
    if cm is None or len(cm.shape) != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("The confusion matrix must be a 2D square matrix.")

    diag_mask = np.eye(cm.shape[0], dtype=bool)
    cmap_diag = 'RdYlGn'
    cmap_off_diag = 'bwr'
    vmax = cm.max()
    norm_off = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=vmax*0.5)
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=vmax*0.1, vmax=vmax*0.9)
    cm_diag = np.ma.masked_array(cm, ~diag_mask)
    cm_off_diag = np.ma.masked_array(cm, diag_mask)

    if ax is None:
        fig, ax = plt.subplots()
    ax.matshow(cm_diag, norm=norm, cmap=cmap_diag, alpha=0.9)
    ax.matshow(cm_off_diag, norm=norm_off, cmap=cmap_off_diag, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    if show_plot:
        plt.show()


def feature_plot(
        X: NDArray,
        y: NDArray,
        feature_names: NDArray,
        y_labels: NDArray,
        title: str = "Feature Plot",
        fig_size: int = 10,
        n_ticks: int = 5,
        show_plot: bool = True,
        fig: Optional[plt.Figure] = None
) -> None:
    """Plot the feature distribution.

    Parameters
    ----------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The class labels.
    feature_names : np.ndarray
        The feature names.
    y_labels : np.ndarray
        The class labels.
    title : str, optional
        The title of the figure. 
    fig_size : int, optional
        The size of the quadratic figure.
    n_ticks : int, optional
        The number of ticks on the axes.
    show_plot : bool, optional
        Whether to show the plot. Default is True.
    fig : plt.Figure, optional
        The figure to plot on. If None, a new figure is created.

    Notes
    -----
    - The diagonal plots show the distribution of the features for each class.
    - The off-diagonal plots show the scatter plot of the features.
    - The color of the scatter plot points corresponds to the class labels.
    - The plot layout is based on the Seaborn pairplot 
      (https://seaborn.pydata.org/generated/seaborn.pairplot.html)
    """
    n_features = X.shape[1]
    n_classes = len(y_labels)
    if n_features != len(feature_names):
        raise ValueError(
            "The number of feature names must match the number of features.")
    if n_classes != len(np.unique(y)):
        raise ValueError(
            "The number of class labels must match the number " +
            "of unique labels in y."
        )
    cmap = DiscreteColorMap()
    colors = [cmap[i] for i in range(n_classes)]
    cmap_listed = mcolors.ListedColormap(colors)
    fig = fig if fig is not None else plt.figure()
    fig.set_size_inches(fig_size, fig_size)
    axs = fig.subplots(n_features, n_features)
    for r in range(n_features):
        for c in range(n_features):
            ax = axs[r, c]
            if r == c:
                for y_i in range(n_classes):
                    ax.hist(X[y == y_i, c], bins=20,
                            color=colors[y_i], alpha=0.5)
            else:
                ax.scatter(X[:, c], X[:, r], c=y, cmap=cmap_listed, alpha=0.7, s=13)
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()
            y_ticks = np.linspace(*y_lim, n_ticks)
            y_tick_lbls = [_num_significant_figures(y) for y in y_ticks]
            x_ticks = np.linspace(*x_lim, n_ticks)
            x_tick_lbls = [_num_significant_figures(x) for x in x_ticks]
            ax.set_yticks(y_ticks, y_tick_lbls)
            ax.set_xticks(x_ticks, x_tick_lbls)
            if r == 0 and c == 1:
                axs[0, 0].set_yticklabels(y_tick_lbls)
            if r == n_features - 1 and c == n_features - 1:
                ax.set_yticklabels(axs[-2, -1].get_yticklabels())
            if r == n_features - 1:
                ax.set_xlabel(feature_names[c])
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(feature_names[r])
            else:
                ax.set_yticklabels([])

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)

    legend = fig.legend(
        handles=[
            plt.Line2D([0], [0], marker='o', color='w', label=y_labels[i],
                    markerfacecolor=colors[i], markersize=10) 
            for i in range(n_classes)
        ],
        loc='center right'
    )
    legend_bbox = legend.get_window_extent(renderer=fig.canvas.get_renderer())
    legend_frac_width = legend_bbox.width / (fig_size * fig.dpi) 
    fig.subplots_adjust(right=1-legend_frac_width*1.1)
    if show_plot:
        plt.show()

def _num_significant_figures(num, n=2, return_str=True):
    """Return a number with n significant figures.
    
    Parameters
    ----------
    num : float
        The number to format.
    n : int, optional
        The number of significant figures. Default is 2.
    return_str : bool, optional
        Whether to return a string. Default is True.
    
    Returns
    -------
    num : float | int | str
        The number with n significant figures.
    """
    sci_str = f"{num:.{n}g}"
    num = float(sci_str)
    if num.is_integer():
        num = int(num)
    if return_str:
        return str(num)
    return num