"""Abstract base class for machine learning models.

This submodule defines the abstract base class for machine learning models
and a abstract base class for hyperparameters. All machine learning models
in this package adhere to the MLModel and MLHyperparams interfaces.


Classes
-------
MLHyperparams
    Abstract base class for hyperparameters of machine learning models.
MLModel
    Abstract base class for machine learning models.
    
Notes
-----
- The API provides a consistent interface for different models, allowing them
  to be measured and compared in a consistent way. 
- The data_tools package uses this interface to train and evaluate different
  machine learning models.
"""


from abc import ABC, abstractmethod
from typing import Any
from numpy.typing import NDArray


class MLHyperparams(ABC):
    """Abstract base class for hyperparameters of machine learning models.
    
    Methods
    -------
    to_dict
        Return hyperparameters as a dictionary.
    __str__
        Return a string representation of the hyperparameters.
    """
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Return hyperparameters as a dictionary."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of hyperparameters."""
        pass


class MLModel(ABC):
    """Abstract base class for machine learning models.
    
    Methods
    -------
    __init__
        Initialize machine learning model.
    fit
        Fit the model to the training data.
    predict
        Predict class labels for new data.
    set_params
        Set hyperparameters for model.

    Properties
    ----------
    hyperparams
        Return hyperparameters for the model.
    """
    
    @abstractmethod
    def __init__(self, hyperparameters: Any) -> None:
        """Initialize machine learning model.
        
        Parameters
        ----------
        hyperparameters: Any
            Hyperparameters for the model
        """
        self._hyperparams = hyperparameters
    
    @property
    @abstractmethod
    def hyperparams(self) -> Any:
        """Return hyperparameters for the model."""
        pass
    
    @abstractmethod
    def fit(self, X: NDArray, y: NDArray) -> None:
        """Fit the model to the training data.
        
        Parameters
        ----------
        X: NDArray
            Feature matrix
        y: NDArray
            Class label vector
        """
        pass
    
    @abstractmethod
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
        pass

    @abstractmethod
    def get_model_props(self, include_hyperparams: bool = True) -> dict:
        """Return properties of the model.
        
        Parameters
        ----------
        include_hyperparams: bool, default=True
            Whether to include hyperparameters in the properties.
        
        Returns
        -------
        props: dict
            Properties of the model
        """
        pass
    
    @abstractmethod
    def set_params(self, **kwargs) -> None:
        """Set hyperparameters for model.

        Parameters
        ----------
        **kwargs: dict
            Hyperparameters to set.

        Raises
        ------
        ValueError
            If an unsupported hyperparameter is provided.
        """
        pass

