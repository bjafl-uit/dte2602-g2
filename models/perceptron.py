"""Perceptron model for binary classification.

This module contains a simple Perceptron class for linear classification.
The Perceptron class is a binary classifier, and will only work for binary
questions.

Classes
---------------
PerceptronHyperparameters
    Hyperparameters for the Perceptron model.
Perceptron
    Perceptron class for binary classification.

Methods
-------
scatter_plot_decision_boundaries
    Plot 2-feature data with decision boundary, or multiple boundaries for
    several perceptrons trained on the same data.

Notes
-----
- See the perceptron_ova module for a One-vs-All Perceptron classifier,
    utilizing the Perceptron class for multi-class classification.
"""
import numpy as np
from dataclasses import dataclass
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .ml_model import MLModel, MLHyperparams
from data_tools.prepare import join_words

@dataclass
class PerceptronHyperparams(MLHyperparams):
    """Hyperparameters for the Perceptron model.

    Attributes
    ----------
    learning_rate: float
        Learning rate for updating weights and bias.
    max_epochs: int
        Maximum number of epochs for training.
    accuracy_goal: float
        Goal for accuracy in training.
    """

    learning_rate: float = 0.01
    max_epochs: int = 200
    accuracy_goal: float = 1.0
    
    def __str__(self) -> str:
        """Return string representation of hyperparameters."""
        return (
            f"Perceptron hyperparameters:\n"
            f"  Learning rate: {self.learning_rate}\n"
            f"  Max epochs: {self.max_epochs}\n"
            f"  Accuracy goal: {self.accuracy_goal}"
        )
    
    def to_dict(self) -> dict:
        """Return hyperparameters as dictionary."""
        return {
            'learning_rate': float(self.learning_rate),
            'max_epochs': int(self.max_epochs),
            'accuracy_goal': float(self.accuracy_goal)
        }

class Perceptron(MLModel):
    """Perceptron class for binary classification.

    Properties
    ----------
    n_features: int
        Number of features in input data (number of perceptron inputs).
    weights: NDArray
        Perceptron weights.
    bias: float
        Perceptron bias value.
    converged: bool
        Boolean indicating if Perceptron has converged during training.

    Class Constants
    ---------------
    RND_WEIGHT_RNG: tuple[float, float]
        Range for random weight initialization.
    BIAS_INIT: float
        Default initial bias value.
    THRESHOLD: float
        Decision threshold for perceptron output.

    Methods
    -------
    predict_single(x: NDArray) -> int
        Calculate perceptron output for a single observation vector.
    predict(X: NDArray) -> NDArray
        Calculate perceptron output for data matrix.
    train(X: NDArray, y: NDArray, learning_rate: float, max_epochs: int)
        Fit perceptron to training data.
    decision_boundary_slope_intercept() -> tuple[float, float]
        Calculate slope and intercept for decision boundary line.
        Only for 2-feature data.
    """

    RND_WEIGHT_RNG = (-1, 1)
    BIAS_INIT = 0
    THRESHOLD = 0

    def __init__(
            self,
            n_features: int,
            hyperparameters: PerceptronHyperparams = None,
            weights: Optional[NDArray] = None,
            bias: Optional[float] = None
    ) -> None:
        """Initialize the perceptron.

        Parameters
        ----------
        n_features: int
            Number of features in input data (number of perceptron inputs).
        weights: NDArray, optional
            Initial weights for the perceptron, shape (n_features,).
            Defaults to random values in range RND_WEIGHT_RNG.
        bias: float, optional
            Initial bias for the perceptron.
            Defaults to BIAS_INIT.
        """
        if n_features < 1:
            raise ValueError("n_features must be positive integer")
        if weights is not None and len(weights) != n_features:
            raise ValueError("Number of weights must match n_features")
        if hyperparameters is None:
            self._hyperparams = PerceptronHyperparams()
        else:
            self._hyperparams = hyperparameters
        self._weights = weights or \
            np.random.uniform(*self.RND_WEIGHT_RNG, size=n_features)  # TODO: sqrt k ?
        self._bias = bias or self.BIAS_INIT
        self._converged = False


    
    @property
    def hyperparams(self) -> PerceptronHyperparams:
        """Return hyperparameters for the model."""
        return self._hyperparams
    
    @property
    def n_features(self) -> int:
        """Number of features in input data.

        Returns
        -------
        int
            Number of features in input data (number of perceptron inputs).
        """
        return len(self._weights)

    @property
    def weights(self) -> NDArray:
        """Perceptron weights.

        Returns
        -------
        NDArray
            Array, shape (n_features,), with perceptron weights.
        """
        return self._weights.copy()

    @property
    def bias(self) -> float:
        """Perceptron bias value.

        Returns
        -------
        float
            Perceptron bias value.
        """
        return self._bias

    @property
    def converged(self) -> bool:
        """Boolean indicating if Perceptron has converged during training.

        Returns
        -------
        bool
            True if Perceptron properties has converged.
        """
        return self._converged

    def get_model_props(self, include_hyperparams: bool = False) -> dict:
        """Return model properties as dictionary.
        
        Parameters
        ----------
        include_hyperparams: bool, optional
            If True, include hyperparameters in the dictionary.
            
        Returns
        -------
        props : dict
            Model properties as dictionary.
        """
        props =  {
            'n_features': self.n_features,
            'weights': [float(w) for w in self.weights],
            'bias': float(self.bias),
            'converged': self.converged
        }
        if include_hyperparams:
            props['hyperparameters'] = self.hyperparams.to_dict()
        return props
    
    def set_params(self, **kwargs) -> None:
        """Set hyperparameters for model.

        Parameters
        ----------
        **kwargs: dict
            Hyperparameters to set. Supported hyperparameters:
            - learning_rate: float
            - max_epochs: int
            - accuracy_goal: float

        Raises
        ------
        ValueError
            If any hyperparameter is invalid
        """ 
        for key, value in kwargs.items():
            if not hasattr(self._hyperparams, key):
                raise ValueError(f"Unsupported hyperparameter: {key}")
            setattr(self._hyperparams, key, value)
        
    def predict_single(self, x: NDArray) -> int:
        """Calculate perceptron output for a single observation vector.

        Parameters
        ----------
        x: NDArray
            Observation vector, shape (n_features,)

        Returns
        -------
        int
            Perceptron output, 1 or 0

        Raises
        ------
        ValueError
            If input vector x does not have same length as weights

        Notes
        -----
        - First the weighted sum of inputs is calculated, given by the formula:
            I = sum(w_i * x_i) + b, for all i in n_features.
        - The output is 1 if I > THRESHOLD, otherwise 0.
        """
        if len(x) != len(self._weights):
            raise ValueError("Input vector x must have same length as weights")
        return 1 if self._weights @ x + self._bias > self.THRESHOLD else 0

    def predict(self, X: NDArray) -> NDArray:
        """Calculate perceptron output for data matrix.

        Parameters
        ----------
        X: NDArray
            Feature matrix, shape (n_samples, n_features).
            Rows are samples of observation vectors.

        Returns
        -------
        NDArray
            Outputs for each row in data matrix X, shape (n_samples,)

        Raises
        ------
        ValueError
            If number of features (columns) in data matrix X is different from
            the perceptrons number of inputs (weights).

        Notes
        -----
        - The perceptron output is calculated for each observation
            vector x in the data matrix X.
        """
        return np.array([self.predict_single(x) for x in X])

    def fit(
            self,
            X: NDArray,
            y: NDArray,
    ) -> None:
        """Fit perceptron to training data X with binary labels y.

        Parameters
        ----------
        X: NDArray
            Training data - feature matrix, shape (n_samples, n_features)
        y: NDArray
            Traning data - class labels, shape (n_samples,)
        learning_rate: float
            Learning rate for updating weights and bias
        max_epochs: int
            Maximum number of epochs for training

        Notes
        -----
        - The perceptron is trained using the perceptron learning algorithm.
        - The perceptron is trained until convergence, max_epochs is
            reached or until accuracy_goal is reached.
        - The perceptron is trained using the following update rules, applied
            for each set x_i, y_i given by the supplied training data:
              w_j = w_j + learning_rate * (y - y_predicted) * x ,
                for all j in n_features.
              b = b + learning_rate * (y - y_predicted)
        - The perceptron is converged if all predictions are correct.
        - Since the weights may jump around when difficult to reach convergence,
          the best weights are saved and used as the final weights.
        """
        self._converged = False
        learning_rate = self._hyperparams.learning_rate
        max_epochs = self._hyperparams.max_epochs
        accuracy_goal = self._hyperparams.accuracy_goal  #TODO
        accuracy = 0
        epoch = 0
        best_epoch = (epoch, accuracy, (self._weights, self.bias))
        while (not self._converged
               and accuracy < accuracy_goal
               and epoch < max_epochs):
            self._converged = True
            totE = 0
            for x_i, y_i in zip(X, y):
                E = y_i - self.predict_single(x_i)
                totE += abs(E)
                if abs(E) > 0:
                    self._weights += learning_rate * E * x_i
                    self._bias += learning_rate * E
                    self._converged = False
            accuracy = 1 - totE / len(y)
            if accuracy > best_epoch[1]:
                best_epoch = (
                    epoch,
                    accuracy,
                    (self._weights.copy(), self._bias)
                )
            epoch += 1
        self._weights, self._bias = best_epoch[2]

    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line.

        Returns
        -------
        slope: float
            Slope of decision boundary line
        intercept: float
            Intercept of decision boundary line

        Raises
        ------
        ValueError
            If weights do not have length 2
        ValueError
            If slope is undefined. When the second weight is zero, the slope 
            is undefined (division by zero).

        Notes
        -----
        - May only be calculated for 2-feature data, where the second weight
            can't be zero (division by zero).
        - The decision boundary line separates the two classes, where the
            perceptron output is 0.
        - For a 2-feature perceptron, the decision boundary line is a straight
            line in the 2D feature space, given by:
              w0 * x0 + w1 * x1 + b = 0
        - This method maps the boundary line to the x-y plane, where the first
            feature is the x-axis and the second feature is the y-axis. The
            line may then be expressed as:
              f(x0) = slope * x0 + intercept.
            From the equation above, we get:
              f(x0) = (-w0 / w1) * x0 + (-b / w1)
        """
        if len(self._weights) != 2:
            raise ValueError("Only for 2-feature data")
        if self._weights[1] == 0:
            raise ValueError("Division by zero: slope undefined")
        slope = -self._weights[0] / self._weights[1]
        intercept = -self._bias / self._weights[1]
        return slope, intercept


def scatter_plot_decision_boundaries(
        X: NDArray,
        y: NDArray,
        decision_boundaries: list[tuple[float, float]] | tuple[float, float],
        y_labels: list[str] = ['Group 1', 'Group 2'],
        y_group1: list[int] = [0],
        axis_labels: list[str] = ['Feature 1', 'Feature 2'],
        accuracy: Optional[ArrayLike | float] = None,
        show_plot: bool = True,
        save_as: str | None = None
) -> None:
    """Plot 2-feature data with decision boundaries.

    Parameters
    ----------
    X: NDArray
        Feature matrix, shape (n_samples, 2)
    y: NDArray
        Class labels, shape (n_samples,)
    decision_boundaries: list of tuples or tuple
        List of decision boundaries as (slope, intercept) tuples.
        If only one boundary, may be provided as a single tuple.
    y_labels: list of str, optional
        String labels for classes in y. Defaults to ['Group 1', 'Group 2'].
        The indices correspond to the class value in y.
    y_group1: list of str or int, optional
        Values for classes grouped in group1. Defaults to [1].
    axis_labels: list of str, optional
        Labels for x and y axes. Defaults to ['Feature 1', 'Feature 2'].
    accuracy: float or list of floats, optional
        Accuracy of decision boundaries. Must match length of
        decision_boundaries. If only one decision boundary, may be given
        as a single float.
    show_plot: bool, optional
        If True, plot is shown. If False, plot is not shown.
    save_as: str, optional
        If provided, plot is saved to file with the given name.

    Raises
    ------
    ValueError
        - If X and y do not have same number of samples
        - If X does not have 2 features
        - If length of accuracy does not match number of decision_boundaries
        - If y_labels does not match unique values in y
    """
    # Validate input
    if isinstance(accuracy, float):
        accuracy = [accuracy]
    if isinstance(decision_boundaries, tuple):
        decision_boundaries = [decision_boundaries]
    y_unique = np.unique(y)
    if X.shape[1] != 2:
        raise ValueError("Only for 2-feature data")
    if X.shape[0] != y.size:
        raise ValueError("X and y must have same number of samples")
    if accuracy is not None and len(accuracy) != len(decision_boundaries):
        raise ValueError(
            "Length of accuracy must equal number of decision_boundaries")
    if len(y_unique) != len(y_labels):
        raise ValueError("Provided y_labels doesn't match unique values in y")

    # Prepare plot data
    grid = np.array(
        [X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()]
    ) * 1.1
    group1_lbls = [y_labels[i] for i in y_group1]
    group0_lbls = [lbl for lbl in y_labels if lbl not in group1_lbls]
    group_names = [join_words(group0_lbls), join_words(group1_lbls)]
    group_sizes = (len(group0_lbls), len(group1_lbls))
    (group_colors,
     cmap_contour) = _plot_generate_cmaps(group_sizes)
    plot_title = _plot_title_db(len(decision_boundaries), accuracy)

    # Plot
    fig = plt.figure(figsize=(8, 6))
    _plot_decision_boundaries(grid, decision_boundaries,
                              accuracy, cmap_contour)
    _plot_observations(X, y, group_colors, y_labels, y_group1)

    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(plot_title)
    plt.axis(grid)
    plt.legend()

    # Show/save plot
    if save_as is not None:
        plt.savefig(save_as)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_title_db(
        n_dbs: int,
        accuracy: Optional[ArrayLike]
) -> str:
    """Generate plot title for decision boundary plot.

    Parameters
    ----------
    n_dbs: int
        Number of decision boundaries
    accuracy: list of floats, optional
        Accuracy of decision boundaries.

    Returns
    -------
    title: str
        Plot title.
    """
    accuracy = np.array(accuracy) if accuracy is not None else None
    if n_dbs > 1:
        title = f'Decision Boundaries plot (n={n_dbs})'
        if accuracy is not None:
            title += ' | Accuracy: '
            title += f'{accuracy.min():.2f} (min) '
            title += f'{accuracy.mean():.2f} (avg) '
            title += f'{accuracy.max():.2f} (best)'
    else:
        title = 'Decision Boundary plot'
        if accuracy is not None:
            title += ' | Accuracy: '
            title += f'{accuracy[0]:.2f}'
    return title


def _plot_decision_boundaries(
        grid: tuple[float, float, float, float],
        dbs: list[tuple[float, float]] | tuple[float, float],
        accuracy: Optional[ArrayLike],
        cmap_contour: mcolors.ListedColormap
) -> None:
    """Plot decision boundaries and contour.

    Parameters
    ----------
    grid: tuple
        Plot grid limits (x_min, x_max, y_min, y_max)
    dbs: list of tuples or tuple
        List of decision boundaries as (slope, intercept) tuples.
        If only one boundary, may be provided as a single tuple.
    accuracy: float or list of floats, optional
        Accuracy of decision boundaries. If list, must match length of
        decision_boundaries. May be given as (single) float, if there
        is only one decision boundary to be plotted.
    cmap_contour: ListedColormap
        Colormap for decision contour plot
    """
    (x_min, x_max, y_min, y_max) = grid
    x_space = np.linspace(x_min, x_max, 500)
    y_space = np.linspace(y_min, y_max, 500)
    x_mesh, y_mesh = np.meshgrid(x_space, y_space)

    # Decision boundaries (db)
    if isinstance(dbs, tuple) and \
       not isinstance(dbs[0], tuple):
        dbs = [dbs]
    db_best = dbs[0]
    if accuracy is not None:
        accuracy = np.array(accuracy)
        db_best = dbs[accuracy.argmax()]
    y_db_best_line = db_best[0] * x_space + db_best[1]

    y_dbs = []
    for slope, intercept in dbs:
        y_dbs.append(slope * x_mesh + intercept)
    mask_below_all = np.logical_and.reduce([y_mesh >= db for db in y_dbs])
    mask_above_all = np.logical_and.reduce([y_mesh <= db for db in y_dbs])
    mask_between = np.logical_not(
        np.logical_or(mask_below_all, mask_above_all)
    )
    boundary_area = np.zeros_like(x_mesh)
    boundary_area[mask_above_all] = 1
    boundary_area[mask_below_all] = 2
    boundary_area[mask_between] = 3

    # Plot decision boundaries and contour
    plt.contourf(
        x_mesh,
        y_mesh,
        boundary_area,
        levels=[0, 1, 2, 3],
        alpha=0.2,
        cmap=cmap_contour,
        zorder=0
    )
    if len(dbs) > 1:
        y_boundary_lines = [a * x_space + b for a, b in dbs]
        last_boundary_line = y_boundary_lines.pop()
        plt.plot(x_space, last_boundary_line, color='black',
                 zorder=1, linewidth=0.1, label='Boundaries')
        for y_db in y_boundary_lines:
            plt.plot(x_space, y_db, color='black', zorder=1, linewidth=0.1)
        plt.plot(x_space, y_db_best_line, color='red',
                 linestyle='--', zorder=1, label='Best')
    else:
        plt.plot(x_space, y_db_best_line, color='black',
                 linewidth=2, zorder=1, label='Boundary')


def _plot_generate_cmaps(
        group_sizes: tuple[int, int],
        cmap0: Union[mcolors.Colormap, str] = 'Reds',
        cmap1: Union[mcolors.Colormap, str] = 'Blues',
        cmap_from: float = 0.9,
        cmap_to: float = 0.5,
        cmap_min_samples: int = 3,
        color_countour_between: str = 'gray'
) -> tuple[NDArray, NDArray, mcolors.ListedColormap]:
    """Get colors for groups and decision contour.

    Parameters
    ----------
    group_sizes: tuple
        Number of samples in group 0 and group 1
    cmap0: str or Colormap, optional
        Colormap for group 0. Defaults to 'Reds'.
    cmap1: str or Colormap, optional
        Colormap for group 1. Defaults to 'Blues'.
    cmap_from: float, optional
        Start color for colormap. Defaults to 0.9.
    cmap_to: float, optional
        End color for colormap. Defaults to 0.5.
    cmap_min_samples: int, optional
        Minimum number of colors sampled. Defaults to 3.
    color_countour_between: str, optional
        Color for decision contour between boundaries. Defaults to 'gray'.
        Value must be a valid color string, recognized by matplotlib.colors.

    Returns
    -------
    group_colors: tuple[NDArray, NDArray]
        Colors for group 0 and group 1.
    cmap_contour: ListedColormap
        Colormap for decision contour plot.

    Notes
    -----
    - If cmap is a string, the corresponding colormap from
      matplotlib.colors is used.
    - If the minimum number of samples is too low, the contrast
      between colors in the same group may be too harsh.
    """
    if isinstance(cmap0, str):
        cmap0 = mpl.colormaps[cmap0]
    if isinstance(cmap1, str):
        cmap1 = mpl.colormaps[cmap1]
    n_gr_a, n_gr_b = group_sizes
    n_color0 = max(n_gr_a, cmap_min_samples)
    n_color1 = max(n_gr_b, cmap_min_samples)
    colors0 = cmap0(np.linspace(cmap_from, cmap_to, n_color0))
    colors1 = cmap1(np.linspace(cmap_from, cmap_to, n_color1))
    cmap_contour = mcolors.ListedColormap([colors0[0], colors1[0],
                                           color_countour_between])
    group_colors = (colors0[:n_color0], colors1[:n_color1])
    return group_colors, cmap_contour


def _plot_observations(X, y, group_colors, y_labels, y_group1):
    """Scatter plot observations with colors for each group."""
    color0, color1 = [iter(c) for c in group_colors]
    for cls_id in np.unique(y):
        row_mask = y == cls_id
        cls_name = y_labels[cls_id]
        color = next(color1) if cls_id in y_group1 else next(color0)
        plt.scatter(X[row_mask, 0], X[row_mask, 1], color=color,
                    label=cls_name, edgecolors='k', zorder=2)
