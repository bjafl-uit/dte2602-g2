"""One-vs-All Perceptron classifier.

This module contains the One-vs-All Perceptron classifier, which is a simple
one-layer ANN with no hidden layers. The model uses the One-vs-All strategy to
handle multi-class classification, where a separate perceptron is trained for
each class.

Classes
-------
PerceptronOVAClassifier
    One-vs-All Perceptron classifier.

Notes
-----
- The model uses the Perceptron class to implement the individual nodes.
"""
import numpy as np
from typing import Optional
from numpy.typing import NDArray

from .perceptron import Perceptron, PerceptronHyperparams
from .ml_model import MLModel

class PerceptronOVAClassifier(MLModel):
    """One-vs-All Perceptron classifier.
    
    Methods
    -------
    fit
        Fit the model to the training data.
    predict
        Predict class labels for new data.
    get_decision_boundaries
        Return decision boundaries for each perceptron.
    set_params
        Set hyperparameters for model.
    
    Properties
    ----------
    hyperparams
        Return hyperparameters for the model.
    n_inputs
        Return number of input features.
    n_classes
        Return number of classes.
    perceptrons
        Return list of perceptrons in the classifier.
    
    Notes
    -----
    - The model acts as a simple one-layer ANN with no hidden layers.
    - The model uses the One-vs-All strategy to handle multi-class 
      classification, where a separate perceptron is trained for each class.
    - The individual nodes are instances of the Perceptron class.
    """

    def __init__(
            self, 
            n_features: int, 
            n_classes: int = -1, 
            hyperparameters: Optional[PerceptronHyperparams] = None 
    ) -> None:
        """Initialize One-vs-All Perceptron classifier.

        Parameters
        ----------
        n_features: int
            Number of features
        n_classes: int, optional (default=-1)
            Number of classes.
        hyperparameters: PerceptronHyperparameters, optional (default=None)
            Hyperparameters for the model

        Notes
        -----
        - If n_classes is not provided, it will be inferred from the data when
            calling the fit method.
        """
        self._n_inputs = n_features
        self._n_classes = n_classes
        if hyperparameters is None:
            hyperparameters = PerceptronHyperparams()
        self._hyperparams = hyperparameters
        self._perceptrons = []
        if n_classes > 0:
            self._init_perceptrons()

    @property
    def hyperparams(self) -> PerceptronHyperparams:
        """Return hyperparameters for the model."""
        return self._hyperparams
    
    @property
    def n_inputs(self) -> int:
        """Return number of input features."""
        return self._n_inputs
    
    @property
    def n_classes(self) -> int:
        """Return number of classes."""
        return self._n_classes
    
    @property
    def perceptrons(self) -> list[Perceptron]:
        """Return list of perceptrons in the classifier."""
        return self._perceptrons
    
    def get_model_props(self, include_hyperparams: bool = False) -> dict:
        """Return properties of the model.
        
        Parameters
        ----------
        include_hyperparams: bool, optional (default=False)
            Whether to include hyperparameters in the output.
        
        Returns
        -------
        dict
            Properties of the model.
        """
        props = {
            'n_inputs': self._n_inputs,
            'n_classes': self._n_classes
        }
        if include_hyperparams:
            props['hyperparameters'] = self._hyperparams.to_dict()
        return props

    def set_params(self, **params) -> None:
        """Set hyperparameters for model.
        
        Parameters
        ----------
        **params: dict
            Hyperparameters to set.
        
        Notes
        -----
        - The hyperparameters must be attributes of the dataclass 
          PerceptronHyperparameters.
        - The hyperparameters are set for all perceptrons in the classifier.
        """
        for param, value in params.items():
            setattr(self._hyperparams, param, value)

    def _init_perceptrons(self) -> None:
        """Initialize the perceptrons."""
        self._perceptrons = [Perceptron(self._n_inputs, self._hyperparams) 
                             for _ in range(self._n_classes)]
        
    def fit(self, X: NDArray, y: NDArray) -> None:
        """Fit the model to the training data.

        Parameters
        ----------
        X: NDArray
            Feature matrix
        y: NDArray
            Class label vector
        """
        if self._n_classes < 0:
            self._n_classes = len(np.unique(y))
            self._init_perceptrons()
            
        for i, perceptron in enumerate(self._perceptrons):
            y_binary = (y == i).astype(int)
            perceptron._hyperparams = self._hyperparams
            perceptron.fit(X, y_binary)
        
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for new data.

        Parameters
        ----------
        X: NDArray
            Feature matrix

        Returns
        -------
        y_pred: NDArray
            Predicted class labels
        """
        predictions = np.zeros((X.shape[0], self._n_classes))
        for i, perceptron in enumerate(self._perceptrons):
            predictions[:, i] = perceptron.predict(X)
        return predictions.argmax(axis=1)
    
    def get_decision_boundaries(self) -> list[tuple[float, float]]:
        """Return decision boundaries for each perceptron.

        Returns
        -------
        list[tuple[float, float]]
            List of decision boundaries for each perceptron.

        Raises
        ------
        ValueError
            If the models perceptrons has not been trained.
        ValueError
            If using more than two features as input.
        """
        if not self._perceptrons:
            raise ValueError("Model has not been initialized.")
        return [perceptron.decision_boundary_slope_intercept() 
                for perceptron in self._perceptrons]
    

    


    