"""Hyperparameter search algorithms.

This module provides classes for hyperparameter search algorithms. The
HyperparameterSearch class is a base class for hyperparameter search algorithms,
and provides a consistent interface for different search algorithms.

Classes
-------
HyperparameterSearch
    Base class for hyperparameter search algorithms.
GridSearch(HyperparameterSearch)
    Grid search for hyperparameter optimization.
RandomSearch(HyperparameterSearch)
    Random search for hyperparameter optimization.

Notes
-----
- The search algorithms here are implemented to work with models in the models
    package, adhering til the MLModel interface.
- This module uses the DataSplitter class from the data_tools package to generate
    train-test splits for the data.
- Scoring metrics are primarly provided through the measure module in this package, 
    but may also be provided as custom user-defined functions. Such functions must 
    take the true and predicted labels as input, and return a float score.
"""
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Iterable, Literal, Optional, Union


from models import MLModel
from .data_splitter import DataSplitter
from .plot import plot_confusion_matrix
from .measure import mean_accuracy, precision_recall, prediction_score, \
     confusion_matrix


class HyperparameterSearch:
    """Base class for hyperparameter search algorithms.
    
    This class provides a base for hyperparameter search algorithms, providing
    a consistent interface for different search algorithms. The class contains
    methods for fitting a model using hyperparameter search, and for evaluating
    the search results.
    """

    def __init__(
            self,
            model: MLModel,
            splitter: DataSplitter,
            scoring: Union[
                Literal['micro', 'macro', 'b-score', 'f-score'], 
                Callable[[NDArray, NDArray], float]
                ] = mean_accuracy,
            model_init_kwargs: dict = {}
    ) -> None:
        """Initialize HyperparameterSearch object.
        
        Parameters
        ----------
        model : MLModel
            The model to fit.
        splitter : DataSplitter
            The data splitter object.
        scoring : callable, default=prediction_accuracy
            The scoring function to use. It can either be a callable function
            that takes the true and predicted labels as input, and returns a
            float, or a string with the following values:
            - 'micro': Calculate the mean accuracy.
            - 'macro': Calculate the precision.
            - 'b-score': Calculate the balanced score.
            - 'f-score': Calculate the frequency score.
        model_init_kwargs : dict, default={}
            The keyword arguments to pass to the model constructor, if needed. 

        Raises
        ------
        ValueError
            If the model is not a subclass or instance of MLModel.
        ValueError
            If the scoring type is invalid.
            
        Notes
        -----
        - During search, if utalizing multithreading, the algorithm creates separate
            instances of the model for each thread. The model must be thread-safe, and 
            necessary init kwargs must be passed to this constructor to successfully
            initialize new model instances.
        - Custom scoring functions must take true labels as the first argument, and
            predicted labels as the second argument. The function must return a float
            score.
        - Setting the scoring parameter to one of the valid strings, uses functions
            from the measure module in this package to calculate the score.
        """
        self._model_init_kwargs = model_init_kwargs
        self._model = model
        if not isinstance(model, MLModel):
            self._model = self._init_new_model()
        self._splitter = splitter
        self._best_params = {}
        self._best_score = None
        self._test_scores = []
        self._test_params = []
        self._X = None
        self._y = None
        self._tot_tests = 0
        self._scoring_args = {}
        if isinstance(scoring, str):
            if scoring == 'micro':
                self._scoring = mean_accuracy
            elif scoring == 'macro':
                self._scoring = precision_recall
                self._scoring_args = {'average': 'macro', 
                                      'return_vals': 'precision'}
            elif scoring in ['b-score', 'f-score']:
                self._scoring = prediction_score
                self._scoring_args = {'score': scoring}
            else:
                raise ValueError("Invalid scoring type.")
        else:
            self._scoring = scoring

    @property
    def best_params(self) -> dict[str, Any]:
        """Get the best hyperparameters from search.

        Returns
        -------
        best_params : dict
            The best hyperparameters found during search.

        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if not self._best_params:
            raise ValueError("The model has not been fitted yet.")
        return self._best_params

    @property
    def best_score(self) -> float:
        """Get the best score from search.
        
        Returns
        -------
        best_score : float
            The best score found during search.
            
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        """
        if self._best_score is None:
            raise ValueError("The model has not been fitted yet.")
        return self._best_score

    def fit(self, X: NDArray, y: NDArray) -> None:
        """Fit the model using the hyperparameter search.

        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        y : np.ndarray
            The labels.

        Notes
        -----
        - This method fits the model using the child class search
          algorithm.
        - The method must be implemented in the child class.
        """
        raise NotImplementedError

    def _multi_thread(self, fit_func: Callable[[int], float], n_threads=0):
        """Handle multithreading for hyperparameter search."""
        if n_threads < 1:
            n_threads = mp.cpu_count()
        n_tests = len(self._test_params)
        with mp.Pool(processes=n_threads) as pool:
            results = pool.map(fit_func, range(n_tests))
        return results

    def _init_new_model(self) -> MLModel:
        """Initialize a new model instance."""
        if isinstance(self._model, type):
            if not issubclass(self._model, MLModel):
                raise ValueError("Model must be a subclass of MLModel.")
            return self._model(**self._model_init_kwargs)
        if isinstance(self._model, MLModel):
            return self._model.__class__(**self._model_init_kwargs)
        raise ValueError("Model must be a subclass or instance of MLModel.")

    def get_test_stats(self) -> NDArray:
        """Get the test scores and hyperparameters.

        Returns
        -------
        stats : np.ndarray
            An array containing the test scores and hyperparameters.
            The last column contains the test scores, and the other columns
            contain the hyperparameter values.
        param_keys : list
            List of the hyperparameter keys, where indices correspond to the
            columns index of the stats array.
        """
        param_keys = sorted(self._best_params.keys())
        stats = np.zeros((len(self._test_scores), len(param_keys) + 1))
        for i, (params, score) in enumerate(zip(self._test_params,
                                                self._test_scores)):
            stats[i, :-1] = [params[key] for key in param_keys]
            stats[i, -1] = score
        return stats, param_keys

    def plot_param_stats(
            self,
            title: str = 'Hyperparameter Search Stats',
            axes: Optional[list[plt.Axes]] = None,
            show_plot: bool = True
            ) -> None:
        """Plot the test scores and hyperparameters.

        Plots the test scores for each hyperparameter value as separate 
        line plots.

        Parameters
        ----------
        title : str
            The title of the plot. Only set if axes is None.
        axes : list of plt.Axes, optional
            The axes to plot the data on. If None, a new figure is created.
        show_plot : bool, default=True
            Whether to show the plot.
        """
        cols, rows, n_params = self._get_col_row_score_plots()
        stats, param_keys = self.get_test_stats()
        cidx_params_to_plot = np.argwhere(np.ptp(stats[:, :-1], axis=0) > 0).ravel()
        if axes is not None:
            axs = axes
        else:
            fig_size = (cols * 4, rows * 4)
            fig, axs = plt.subplots(rows, cols,
                                    figsize=fig_size,
                                    squeeze=False)
        # Get bounds
        y_min = np.min(stats[:, -1])
        y_max = np.max(stats[:, -1])
        if y_min == y_max:
            y_min -= 0.05
            y_max += 0.05
        freq_hists_bins = [np.histogram(stats[:, i], bins=10)
                           for i in cidx_params_to_plot]
        freq_max = np.max([np.max(hist) for hist, _ in freq_hists_bins])
        if freq_max == 0:
            freq_max = 1

        # Make plots for each parameter
        color_score = plt.cm.get_cmap('tab10')(0)
        color_freq = 'gray'
        for i, c_idx in enumerate(cidx_params_to_plot):
            param_name = param_keys[c_idx]
            # Get subplot, add ax for frequency plot
            r, c = divmod(i, cols)
            ax = axs[r, c]
            ax_freq = ax.twinx()

            # Calculate mean, SD and frequency
            x = np.unique(stats[:, c_idx])
            y = np.zeros_like(x)
            sd = np.zeros_like(x)
            for j, val in enumerate(x):
                val_mask = stats[:, c_idx] == val
                y[j] = np.mean(stats[val_mask, -1])
                sd[j] = np.std(stats[val_mask, -1])
            freq_hist, freq_bins = freq_hists_bins[i]

            # Plot data
            ax_freq.stairs(freq_hist, freq_bins, fill=True, 
                           color=color_freq, alpha=0.3)
            ax.plot(x, y, color=color_score)
            ax.fill_between(x, y - sd, y + sd, 
                            color=color_score, alpha=0.2)
            
            # Set labels, limits and legend
            ax.set_xlabel(param_name)
            ax.set_ylim(y_min, y_max)
            ax_freq.set_ylim(0, freq_max)
            if np.allclose(x, x.astype(int)):
                ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
            if i % cols == 0:
                ax.set_ylabel('Score')
            else:
                ax.set_yticklabels([])
            if i % cols == cols - 1:
                ax_freq.set_ylabel('Sample Frequency')
            else:
                ax_freq.set_yticklabels([])
            if i == n_params - 1:
                proxy_freq = plt.Rectangle((0,0), 1, 1, 
                                           facecolor=color_freq, alpha=0.3)
                proxy_sd = plt.Rectangle((0,0), 1, 1, 
                                         facecolor=color_score, alpha=0.2)
                proxy_mean = plt.Line2D([0], [0], color=color_score)
                plots = [proxy_mean, proxy_sd, proxy_freq]
                labels = ['mean', '+/- SD', 'samples']
                ax.legend(plots, labels)
        
        if axes is None:
            fig.suptitle(title)
            fig.tight_layout()
        if show_plot:
            plt.show()

    def plot_confusion_matrices(
            self,
            y_labels: list[str],
            n_best: int = 3,
            title: str = 'Confusion Matrices',
            ax_titles: bool = True,
            axes: Optional[list[plt.Axes]] = None,
            show_plot: bool = True
    ) -> plt.Axes:
        """Plot the confusion matrices for the best models.

        Plots a heatmap of the confusion matrix for the n_best models 
        that achieved the highest scores.

        Parameters
        ----------
        y_labels : list
            The class labels.
        n_best : int, default=3
            The number of best models to plot.
        title : str, default='Confusion Matrices'
            The title of the plot.
        ax_titles : bool, default=True
            Whether to show the subplot titles.
        """
        stats, params = self.get_test_stats()
        best_idxs = np.argsort(-stats[:, -1])[:n_best]
        cols, rows = self._get_col_row_cm_plots(n_best)
        if axes is not None:
            axs = axes
        else:
            fig_size = (cols * 4, rows * 4)
            fig, axs = plt.subplots(rows, cols, figsize=fig_size)

        for i, idx in enumerate(best_idxs):
            r, c = divmod(i, cols)
            ax = axs[r, c]

            best_params = {
                key: stats[idx, i]
                for i, key in enumerate(params)
            }
            self._model.set_params(**best_params)
            (X_train,
             X_test,
             y_train,
             y_test) = self._splitter.split_Xy_data(self._X, self._y)[0]
            self._model.fit(X_train, y_train)
            y_pred = self._model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(cm, y_labels, ax, show_plot=False)
            score = self._scoring(y_test, y_pred, **self._scoring_args)
            title = ""
            if i > 0:
                title += f"{i + 1}. "
            title += "Best"
            ax.set_xlabel('Predicted')
            if i % cols == 0:
                ax.set_ylabel('True')
            else:
                ax.set_ylabel('')
            if ax_titles:
                ax.set_title(title, loc='left', fontsize=11)
                ax.set_title(f"Score {score:.2f}",
                             loc='right', fontsize=8)
            ax.set_title("", loc='center')

        if axes is None:
            fig.suptitle(title)
            fig.tight_layout()
        if show_plot:
            plt.show()
        return axs

    def _get_col_row_cm_plots(self, n_best: int) -> tuple[int, int]:
        """Helper function to define subplot grid size for cm plot."""
        n_cols = min(n_best, 4)
        n_rows = n_best // n_cols
        if n_best % n_cols:
            n_rows += 1
        return n_cols, n_rows

    def _get_col_row_score_plots(self) -> tuple[int, int, int]:
        """Get subplot grid size and n_params to plot for scores plot."""
        params = np.array(list(map(lambda d: list(d.values()), self._test_params)))
        n_params = int(np.sum(np.ptp(params, axis=0) > 0))
        n_cols = min(n_params, 3)
        n_rows = n_params // n_cols
        if n_params % n_cols:
            n_rows += 1
        return n_cols, n_rows, n_params

    def plot_score_cm(
            self,
            y_labels: list[str],
            title_hyperparam_scores: str = 'Hyperparam Scores',
            title_confusion_matrices: str = 'Confusion Matrices',
            fig_score: plt.Figure = None,
            fig_cm: plt.Figure = None,
            show_plot: bool = True,
            kw_cm: dict = {},
            kw_score: dict = {}
    ) -> None:
        """Plot the test scores and confusion matrices.

        Plots the test scores and confusion matrices for the best models,
        in two separate figures.

        Parameters
        ----------
        y_labels : list
            The class labels.
        title_hyperparam_scores : str, default='Hyperparam Scores'
            The title of the hyperparameter scores plot.
        title_confusion_matrices : str, default='Confusion Matrices'
            The title of the confusion matrices plot.
        fig_score : plt.Figure, optional
            The figure to plot the hyperparameter scores.
        fig_cm : plt.Figure, optional
            The figure to plot the confusion matrices.
        show_plot : bool, default=True
            Whether to show the plot.
        kw_cm : dict, optional
            Additional keyword arguments to pass to
            GridSearch.plot_confusion_matrices
        kw_score : dict, optional
            Additional keyword arguments to pass to
            GridSearch.plot_param_stats
        
        Notes
        -----
        - Uses the GridSearch.plot_confusion_matrices and
            GridSearch.plot_param_stats methods.
        - For additional keyword arguments, see the respective methods.
        """
        n_best = kw_cm.pop('n_best', 3)
        c_score, r_score, n_params = self._get_col_row_score_plots()
        c_cm, r_cm = self._get_col_row_cm_plots(n_best)

        fig_score = fig_score if fig_score is not None else plt.figure()
        fig_score.set_size_inches(c_score * 4, r_score * 3.5)
        axs_score = fig_score.subplots(
            r_score, c_score,
            squeeze=False
        )
        fig_cm = fig_cm if fig_cm is not None else plt.figure()
        fig_cm.set_size_inches(c_cm * 3, r_cm * 3)
        axs_cm = fig_cm.subplots(
            r_cm, c_cm,
            squeeze=False
        )
        self.plot_param_stats(
            axes=axs_score,
            show_plot=False,
            **kw_score
        )
        self.plot_confusion_matrices(
            axes=axs_cm,
            y_labels=y_labels,
            n_best=n_best,
            show_plot=False,
            **kw_cm
        )

        if n_params < c_score * r_score:
            for ax in axs_score.ravel()[n_params:]:
                ax.axis('off')
        if n_best < c_cm * r_cm:
            for ax in axs_cm.ravel()[n_best:]:
                ax.axis('off')
                ax.set_title("")
        fig_cm.suptitle(title_confusion_matrices)
        fig_score.suptitle(title_hyperparam_scores)
        fig_score.tight_layout(h_pad=2, w_pad=2)
        fig_cm.tight_layout(h_pad=2, w_pad=2)
        if show_plot:
            plt.show()


class GridSearch(HyperparameterSearch):
    """Grid search for hyperparameter optimization.

    This class performs a grid search over the hyperparameters of a model.
    The model is fitted using all hyperparameter combinations, to find the
    best hyperparameter values.
    """

    def __init__(
            self,
            model,
            param_grid: dict[str, list],
            splitter: DataSplitter,
            scoring: Union[
                Literal['micro', 'macro', 'b-score', 'f-score'], 
                Callable[[NDArray, NDArray], float]
                ] = mean_accuracy
    ) -> None:
        """Initialize GridSearch object.

        Parameters
        ----------
        model : object
            The model to fit.
        param_grid : dict
            The hyperparameter grid to search.
        splitter : DataSplitter
            The data splitter object.
        scoring : callable, default=prediction_accuracy
            The scoring function to use.
            
        Notes
        -----
        - During search, if utalizing multithreading, the algorithm creates separate
            instances of the model for each thread. The model must be thread-safe, and 
            necessary init kwargs must be passed to this constructor to successfully
            initialize new model instances.
        - Custom scoring functions must take true labels as the first argument, and
            predicted labels as the second argument. The function must return a float
            score.
        - Setting the scoring parameter to one of the valid strings, uses functions
            from the measure module in this package to calculate the score.
        """
        super().__init__(model, splitter, scoring)
        self._param_grid = param_grid

    @property
    def n_param_combinations(self) -> int:
        """Get the number of hyperparameter combinations.
        
        Returns
        -------
        n_param_combinations : int
            The number of hyperparameter combinations.
        """
        if self._n_param_combinations is None:
            self._n_param_combinations = self._get_len_param_combinations()
        return self._n_param_combinations

    def _get_len_param_combinations(self) -> int:
        """Helper method to calculate the number of combinations."""
        n_combinations = 1
        for values in self._param_grid.values():
            n_combinations *= len(values)
        return n_combinations

    def fit(
            self,
            X: NDArray,
            y: NDArray
    ) -> dict[str, Any]:
        """Fit the model using GridSearch.

        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        y : np.ndarray
            The labels.

        Notes
        -----
        - This method fits the model using all hyperparameter combinations
        - The best hyperparameters are stored in the best_params property.
        """
        self._X = X
        self._y = y
        self._best_score = -np.inf
        self._test_scores = []
        self._test_params = []
        for params in self._param_combinations():
            self._model.set_params(**params)
            self._test_params.append(params)
            scores = []
            for train_idx, test_idx in self._splitter.split_idxs(X.shape[0]):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                self._model.fit(X_train, y_train)
                y_pred = self._model.predict(X_test)
                scores.append(self._scoring(y_test, y_pred, 
                                            **self._scoring_args))
            mean_score = np.mean(scores)
            self._test_scores.append(mean_score)
            if mean_score > self._best_score:
                self._best_score = mean_score
                self._best_params = params

    def _param_combinations(self) -> list[dict[str, Any]]:
        """Generate all combinations of hyperparameters.

        Returns
        -------
        param_combinations : list of dicts
            List of dictionaries containing hyperparameter combinations.
        """
        param_combinations = [{}]
        self._n_param_combinations = 0
        for param, values in self._param_grid.items():
            new_combinations = []
            for value in values:
                for params in param_combinations:
                    new_params = params.copy()
                    new_params[param] = value
                    new_combinations.append(new_params)
                    self._n_param_combinations += 1
            param_combinations = new_combinations
        return param_combinations


class RandomSearch(HyperparameterSearch):
    """Random search for hyperparameter optimization.

    This class performs a random search over the hyperparameters of a model.
    The model is fitted using a random selection of hyperparameter values, to
    find the best hyperparameter values.

    """

    def __init__(
            self,
            model,
            param_distributions: dict[str, Union[tuple[int, int] | Iterable]],
            splitter: DataSplitter,
            n_iter: int = 100,
            scoring: Union[
                Literal['micro', 'macro', 'b-score', 'f-score'], 
                Callable[[NDArray, NDArray], float]
                ] = mean_accuracy,
            model_init_kwargs: dict = {},
            seed=None
    ) -> None:
        """Initialize RandomSearch object.

        Parameters
        ----------
        model : object
            The model to fit.
        param_distributions : dict
            The hyperparameter distributions to search. The values should either
            be tuples containing the left and right bounds (both inclusive)
            of a integer parameter range to explore, or an Iterable object
            containing the values to explore (e.g. numpy.linspace). If giving a
            single value, be sure to wrap it in a list.
        splitter : DataSplitter
            The data splitter object.
        n_iter : int, default=100
            The number of iterations to perform.
        scoring : callable | Literal, default=prediction_accuracy
            The scoring function to use. It can either be a callable function
            that takes the true and predicted labels as input, and returns a
            float, or a string with the following values:
            - 'micro': Calculate the mean accuracy.
            - 'macro': Calculate the precision.
            - 'b-score': Calculate the balanced score.
            - 'f-score': Calculate the frequency score.
        model_init_kwargs : dict, default={}
            The keyword arguments to pass to the model constructor, if needed. 
        seed : optional
            The random seed to use See numpy.random.default_rng
            for more information.

        Raises
        ------
        ValueError
            If the hyperparameter bounds are given by tuples and are not integers.
        ValueError
            If the left bound is greater than the right bound.
        ValueError
            If the bounds are not given by tuples or Iterable objects.

        Notes
        -----
        - Uses the numpy default random number generator
          (numpy.random.default_rng), BitGenerator (PCG64). See numpy docs.
        - During search, if utalizing multithreading, the algorithm creates separate
            instances of the model for each thread. The model must be thread-safe, and 
            necessary init kwargs must be passed to this constructor to successfully
            initialize new model instances.
        - Custom scoring functions must take true labels as the first argument, and
            predicted labels as the second argument. The function must return a float
            score.
        - Setting the scoring parameter to one of the valid strings, uses functions
            from the measure module in this package to calculate the score.
        """
        for _, bounds in param_distributions.items():
            if isinstance(bounds, tuple): 
                if len(bounds) != 2:
                    raise ValueError(
                        "The bounds must be a tuple with length 2."
                    )
                lbound, rbound = bounds
                if not isinstance(lbound, int) or not isinstance(rbound, int):
                    raise ValueError("The bounds must be integers.")
                if lbound > rbound:
                    raise ValueError("The left bound must be <= to the right.")
            elif not isinstance(bounds, Iterable):
                raise ValueError(
                    "The bounds must be a tuple or an array-like object."
                )
        super().__init__(model, splitter, scoring, model_init_kwargs)
        self._param_distributions = param_distributions
        self._n_iter = n_iter
        self._rng = np.random.default_rng(seed=seed)

    def fit(
            self,
            X: NDArray,
            y: NDArray,
            n_threads: int = 0
    ) -> dict[str, Any]:
        """Fit the model using RandomSearch.

        Parameters
        ----------
        X : np.ndarray
            The data to fit.
        y : np.ndarray
            The labels.
        n_threads : int, default=0
            The number of threads to use. If 0, the number of threads is set
            to the number of CPUs available.

        Returns
        -------
        best_params : dict
            The best hyperparameters found.

        Notes
        -----
        - This method fits the model by randomly selecting hyperparameters
            from the given distributions.
        - The number of iterations is set by the n_iter parameter, and
            determines the number of hyperparameter combinations to test.
        - The best hyperparameters are stored in the best_params property.
        """
        self._X = X
        self._y = y
        self._test_params = []
        n = 0
        skipped = 0
        while n < self._n_iter:
            params = self._random_params()
            if params in self._test_params and skipped < self._n_iter:
                skipped += 1
                continue
            self._test_params.append(params)
            n += 1
        if n_threads == 1:
            test_scores = [self._fit_single(i) for i in range(len(self._test_params))]
        else:
            test_scores = self._multi_thread(self._fit_single, n_threads)
        self._test_scores = test_scores
        best_score_idx = np.argmax(test_scores)
        self._best_score = test_scores[best_score_idx]
        self._best_params = self._test_params[best_score_idx]
        return self._best_params

    def _fit_single(
            self,
            test_param_idx: int) -> float:
        """Fit a model using a single set of hyperparameters."""
        params = self._test_params[test_param_idx]
        model = self._init_new_model()
        model.set_params(**params)
        scores = []
        for train_idx, test_idx in self._splitter.split_idxs(self._X.shape[0]):
            X_train, X_test = self._X[train_idx], self._X[test_idx]
            y_train, y_test = self._y[train_idx], self._y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(self._scoring(y_test, y_pred, **self._scoring_args))

        return np.mean(scores)

    def _random_params(self) -> dict[str, Any]:
        """Generate single set of random hyperparameters."""
        params = {}
        for param, values in self._param_distributions.items():
            if isinstance(values, tuple):
                lbound, rbound = values
                params[param] = self._rng.integers(lbound, rbound + 1)
            else:
                params[param] = self._rng.choice(values)
        return params
