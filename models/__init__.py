"""Package containing machine learning models.

This package contains classes for machine learning models. All models in this package 
adhere to the abstract base class MLModel, which defines the interface for the models.

Modules
-------
ml_model
    Abstract base class for machine learning models.
perceptron
    Perceptron classifier.
perceptron_ova
    One-vs-All Perceptron classifier.
decision_tree
    Decision tree classifier.
decision_tree_nodes
    Node classes for decision tree classifier.
"""

from .ml_model import MLModel
__all__ = ['MLModel']