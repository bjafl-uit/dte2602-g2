"""Decision tree model for classification.

This submodule contains the DecisionTree class, which implements a supervised machine
learning model for classification based on decision trees. The class is a subclass
of the MLModel class, which defines the interface for machine learning models in 
this module.

Classes
-------
DTHyperparams
    Hyperparameters for decision tree model.
DecisionTree
    Decision tree model for classification.

Notes
-----
- The DecisionTree class is a model for classification based on decision trees.
- The model is built using a recursive binary tree structure, where each node
    represents a decision based on a feature value.
- The tree is built by recursively splitting the dataset based on the feature
    that provides the best gini impurity reduction.
- The tree can be visualized using the graphviz library, and the DOT string
    can be generated using the generate_dot method.
""" 

from typing import Optional
from numpy.typing import NDArray
import numpy as np
from dataclasses import dataclass

from .ml_model import MLModel
from .decicion_tree_nodes import DTBranchNode, DTLeafNode, DTNodeStats
from data_tools.color_map import DiscreteColorMap
import data_tools.gini as gini


@dataclass
class DTHyperparams:
    """Hyperparameters for decision tree model.

    Attributes
    ----------
    max_depth: int
        Maximum depth of the decision tree. If None, the tree is grown until
        no further impurity reduction can be achieved.
    min_samples_split: int
        Minimum number of samples required to split an internal node.
    min_samples_leaf: int
        Minimum number of samples required to be at a leaf node.
    post_prune: bool
        If True, the decision tree is post pruned after building.
    """

    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    post_prune: bool = False

    def __post_init__(self):
        """Check hyperparameters for validity."""
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")

    def __str__(self) -> str:
        """Return user friendly string representation of hyperparameters."""
        return f"max_depth={self.max_depth}, " + \
               f"min_samples_split={self.min_samples_split}, " + \
               f"min_samples_leaf={self.min_samples_leaf}, " + \
                f"post_prune={self.post_prune}"
    
    def to_dict(self) -> dict:
        """Return hyperparameters as dictionary."""
        return {
            'max_depth': -1 if self.max_depth is None else int(self.max_depth),
            'min_samples_split': int(self.min_samples_split),
            'min_samples_leaf': int(self.min_samples_leaf),
            'post_prune': self.post_prune
        }

    def evaluate_parent(self, y: NDArray, node_depth: int) -> bool:
        """Evaluate if hyperparameter pruning conditions are met for parent node.

        Parameters
        ----------
        y: NDArray
            NumPy class label vector
        node_depth: int
            Depth of the node in the decision tree

        Returns
        -------
        conditions_met: bool
            True if hyperparameter pruning conditions are met, False otherwise.
        """
        if self.max_depth is not None and node_depth >= self.max_depth:
            return True
        if y.size < self.min_samples_split:
            return True
        return False


class DecisionTree(MLModel):
    """Decision tree model for classification.

    The DecisionTree class implements a supervised machine learning model for
    classification based on decision trees. The model is built using a recursive
    binary tree structure, where each node represents a decision based on a
    feature value. The tree is built by recursively splitting the dataset based
    on the feature that provides the best gini impurity reduction.
    """

    def __init__(
            self,
            X: Optional[NDArray] = None,
            y: Optional[NDArray] = None,
            hyperparameters: Optional[DTHyperparams] = None,
            feature_names: Optional[list[str]] = None,
            y_labels: Optional[list[str]] = None,
            color_map: Optional[DiscreteColorMap] = None
    ) -> None:
        """Initialize decision tree.

        Initializes a new decision tree and fits it to the provided dataset.
        If no parameters are provided, an empty decision tree is created.

        Parameters
        ----------
        X: NDArray, numeric, optional
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, optional
            NumPy class label vector, shape (n_samples,)
        hyperparameters: DTHyperparams, optional
            Hyperparameters for the decision tree. If not provided, no
            pre pruning is applied.
        feature_names: list[str], optional
            List of feature names. Length must match the number of features
            in the dataset, and the order must match the order of the columns
            in the feature matrix.
        y_labels: list[str], optional
            List of class labels. Length must match the number of unique
            classes in the dataset, and the indicies must match the class
            integer ids.
        color_map: DTColorMap, optional
            Color map for visualizing the decision tree. If not provided, the
            default colormap is used (see DTColorMap).

        Raises
        ------
        ValueError
            - If X or y are provided, but not both.
            - If X and y do not have the same number of rows.
            - If X is not a numeric array.
            - If y is not an integer array and cannot be converted to one.
            - If the number of feature names does not match the number of
              features (cols) in X.
            - If the number of class labels does not match the number of unique
              classes (items) in y.
        """
        self._feature_names = feature_names
        self._y_labels = y_labels
        self._root = None
        self._n_features = -1
        self._n_classes = -1
        self._n_nodes = 0
        self._n_nodes_post_pruned = -1
        self._max_depth = 0
        self._color_map = color_map or DiscreteColorMap()
        self._hyperparameters = hyperparameters or DTHyperparams()
        if X is not None or y is not None:
            if y is None or X is None:
                raise ValueError("X and y vector must either " +
                                 "both be provided or None")
            if feature_names and len(feature_names) != X.shape[1]:
                raise ValueError("Number of feature names must match " +
                                 "number of features in X")
            if y_labels and len(y_labels) != len(np.unique(y)):
                raise ValueError("Number of class labels must match " +
                                 "number of unique classes in y")
            self.fit(X, y)

    
    @property
    def hyperparams(self) -> DTHyperparams:
        """Get hyperparameters for decision tree.

        Returns
        -------
        hyperparams: DTHyperparams
            Hyperparameters for decision tree.
        """
        return self._hyperparameters
    
    @property
    def n_nodes(self) -> int:
        """Get number of nodes in decision tree.

        Returns
        -------
        n_nodes: int
            Number of nodes in the decision tree.
        """
        return self._n_nodes

    @property
    def feature_names(self) -> list[str] | None:
        """Get feature names.

        Returns
        -------
        feature_names: list[str] | None
            List of feature names. If not set, returns None.

        Notes
        -----
        - The indices of the feature names corresponds to the tree node's
        feature indexes. This index is the same as the column index of the
        features in the dataset provided when fitting the tree.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names: list[str]):
        """Set feature names.

        Parameters
        ----------
        feature_names: list[str]
            List of feature names. Length must match the number of features
            in the dataset, and the indices must match the column indices of
            the features in the dataset.

        Raises
        ------
        ValueError
            If the number of feature names does not match the number of
            features used in training the tree.

        Notes
        -----
        - The feature names are used for visualizing the decision tree, and
        string representations of the nodes.
        - If the feature names are not set, the feature indices are used.
        """
        if self._n_features != -1 and len(feature_names) != self._n_features:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) " +
                f"does not match number of features ({self._n_features})"
            )
        self._feature_names = feature_names

    @property
    def y_labels(self) -> list[str] | None:
        """Get class labels.

        Returns
        -------
        y_labels: list[str] | None
            List of class labels. If not set, returns None.

        Notes
        -----
        - The indices of the class labels corresponds to the class integer
        ids used in the tree nodes. See also notes for DecisionTree.fit().
        """
        return self._y_labels

    @y_labels.setter
    def y_labels(self, y_labels: list[str]):
        """Set class labels.

        Parameters
        ----------
        y_labels: list[str]
            List of class labels. Length must match the number of unique
            classes in the dataset.

        Raises
        ------
        ValueError
            If the number of class labels does not match the number of
            classes in the dataset used in training the tree.

        Notes
        -----
        - The class labels are used for visualizing the decision tree, and
        string representations of the nodes.
        - If the class labels are not set, the class int ids are used.
        """
        if self._n_classes != -1 and len(y_labels) != self._n_classes:
            raise ValueError(
                f"Number of class labels ({len(y_labels)}) " +
                f"does not match number of classes ({self._n_classes})"
            )
        self._y_labels = y_labels

    @property
    def color_map(self) -> DiscreteColorMap:
        """Get color map for decision tree.
        
        Used for visualizing the decision tree nodes as DOT figure (graphviz).

        Returns
        -------
        color_map: DTColorMap
            Color map for decision tree nodes.
        """
        return self._color_map

    @color_map.setter
    def color_map(self, color_map: DiscreteColorMap):
        """Set color map for decision tree.

        Parameters
        ----------
        color_map: DTColorMap
            Color map for decision tree nodes.
        """
        self._color_map = color_map

    @property
    def n_nodes_post_pruned(self) -> int:
        """Get number of nodes that were pruned during post pruning.
        
        Returns
        -------
        n_nodes_post_pruned: int
            Number of nodes pruned during post pruning.

        Notes
        -----
        - The property returns -1 if the tree has not been pruned. 
            0 means the pruning did not remove any nodes.
        """
        return self._n_nodes_post_pruned

    def get_feature_name(self, feature_index: int) -> str:
        """Get feature name for feature index.

        Parameters
        ----------
        feature_index: int
            Index of feature

        Returns
        -------
        feature_name: str
            Name of feature

        Raises
        ------
        ValueError
            If feature names have not been set or if feature index is out of
            bounds
        """
        if self._feature_names is None:
            raise ValueError("Feature names not set")
        if feature_index < 0 or feature_index >= len(self._feature_names):
            raise ValueError(f"Feature index {feature_index} out of bounds.")
        return self._feature_names[feature_index]

    def get_class_label(self, class_id: int) -> str:
        """Get class label for class index.

        Parameters
        ----------
        y_index: int
            Index of class

        Returns
        -------
        y_label: str
            Class label

        Raises
        ------
        ValueError
            If class labels have not been set or if class index is out of
            bounds
        """
        if self._y_labels is None:
            raise ValueError("Class labels not set")
        if class_id < 0 or class_id >= len(self._y_labels):
            raise ValueError(f"Class index {class_id} out of bounds.")
        return self._y_labels[class_id]

    def get_model_props(self, include_hyperparams: bool = False) -> dict:
        """Get model properties as dictionary.

        Parameters
        ----------
        include_hyperparams: bool
            If True, include hyperparameters in the dictionary.        
        
        Returns
        -------
        model_props: dict
            Dictionary with model properties.
        """
        props = {
            'n_features': self._n_features,
            'n_classes': self._n_classes,
            'n_nodes': self._n_nodes,
            'n_nodes_post_pruned': self._n_nodes_post_pruned
        }
        if include_hyperparams:
            props['hyperparameters'] = self._hyperparameters.to_dict()
        return props
    
    def set_params(self, **kwargs) -> None:
        """Set hyperparameters for decision tree.

        Parameters
        ----------
        **kwargs: dict
            Hyperparameters to set. Supported hyperparameters are:
            - max_depth: int
            - min_samples_split: int
            - min_samples_leaf: int

        Raises
        ------
        ValueError
            If an unsupported hyperparameter is provided.
        """
        for key, value in kwargs.items():
            if key not in DTHyperparams.__annotations__:
                raise ValueError(f"Unsupported hyperparameter: {key}")
            setattr(self._hyperparameters, key, value)

    def fit(self, X: NDArray, y: NDArray):
        """Build and train decision tree based on labelled dataset.

        Parameters
        ----------
        X: NDArray, numeric
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,). Will be converted to
            integers.

        Raises
        ------
        ValueError
            If X and y do not have the same number of rows.
            If X is not a numeric NumPy array.
            If y is not an integer NumPy array and cannot be converted to one

        Notes
        -----
        - If y is non-numeric, the unique values of y are stored in a sorted
        array as string labels. The y-vector is then converted to integers
        based on the index of the sorted unique values.
        - If properties feature_names or y_labels are already set, they will
        be reset to None if the length of the property doesn't match the
        corresponding number of features or classes in X and y.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("X must be a numeric NumPy array")
        if not np.issubdtype(y.dtype, np.integer):
            try:
                y = y.astype(int)
            except ValueError:
                y_unique = np.unique(y)
                self._y_labels = [str(y_val) for y_val in y_unique]
                for i, y_val in enumerate(y_unique):
                    y[y == y_val] = i

        self._n_features = X.shape[1]
        self._n_classes = len(np.unique(y))
        if self._feature_names and \
           len(self._feature_names) != self._n_features:
            self._feature_names = None
        if self._y_labels and len(self._y_labels) != self._n_classes:
            self._y_labels = None

        self._n_nodes = 0
        self._max_depth = 0
        self._root = self._build_tree(X, y)

        if self._hyperparameters.post_prune:
            self.post_prune()

    def _build_tree(self, X: NDArray, y: NDArray, cur_depth: int = -1):
        """Recursively build decision tree.

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)
        cur_depth: int
            Current depth of the tree (recursion depth).

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no
        impurity reduction can be achieved, a leaf node is created, and its
        value is set to the most common class in y. If a split can achieve
        impurity reduction, a decision (branch) node is created, with left and
        right subtrees created by recursively calling _build_tree on the left
        and right subsets.

        """
        # Update metadata and check hyperparameters
        self._n_nodes += 1
        cur_depth += 1
        self._max_depth = max(self._max_depth, cur_depth)
        pruning_cond_met = self._hyperparameters.evaluate_parent(y, cur_depth)

        # Find best split feature and value
        if not pruning_cond_met:
            (impurity_reduction,
             feature_index,
             feature_value,
             gini_current_node) = gini.best_split_feature_value(
                 X, y, 
                 self._hyperparameters.min_samples_leaf
            )
        else:
            impurity_reduction = 0
            gini_current_node = gini.gini_impurity(y)

        # Get node stats
        samples_per_class = [
            len(y[y == i])
            for i in range(self._n_classes)
        ]
        node_stats = DTNodeStats(gini_current_node, len(y), samples_per_class)
        value = np.argmax(samples_per_class)

        # Generate tree nodes recursively
        if impurity_reduction == 0:
            return DTLeafNode(value, self, node_stats)
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask], cur_depth)
            right = self._build_tree(X[~left_mask], y[~left_mask], cur_depth)
            return DTBranchNode(value, feature_index, feature_value,
                                left, right, self, node_stats)
        
    def post_prune(self) -> int:
        """Post pruning of decision tree.

        Simple pruning that combines sub trees with the same class label to
        leaf nodes. The pruning is done recursively starting from the root
        node. 

        Notes
        -----
        - The pruning is done in place, discarding the pruned nodes.
        - Post pruning happens automatically after the tree is built if the
          hyperparameter post_prune is set to True.

        Returns
        -------
        n_nodes_post_pruned: int
            Number of nodes pruned during post pruning

        Raises
        ------
        ValueError
            If the decision tree is empty (root is None).
        """
        if self._root is None:
            raise ValueError("Decision tree is empty (root is None)")
        self._n_nodes_post_pruned = self._root.post_prune()
        self._n_nodes -= self._n_nodes_post_pruned
        return self._n_nodes_post_pruned


    def predict(self, X: NDArray) -> NDArray:
        """Predict class (y vector) for feature matrix X.

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is None:
            raise ValueError("Decision tree root is None (not set)")
        return self._root.predict(X)

    def generate_dot(
            self,
            fill_colors: bool = True,
            max_nodes: int = 1000
    ) -> str:
        """Generate graphviz figure DOT string of the tree.

        Parameters
        ----------
        fill_colors: bool
            If True, fill the nodes with colors based on the class distribution
            in the node.
        max_nodes: int
            Maximum number of nodes to include in the DOT string. If the tree
            has more nodes, the generation is aborted.

        Returns
        -------
        dot: str
            Graphviz DOT string

        Raises
        ------
        ValueError
            If the root node is not set.
        ValueError
            If the number of nodes in the tree exceeds the max_nodes limit.

        Notes
        -----
        - The DOT string can be visualized using graphviz.Source.
        - If fill_colors is True, the nodes are filled with colors based on the
          class distribution in the node. The color map is defined by the
          color_map property.
        """
        if self._root is None:
            raise ValueError("Decision tree root is None (not set)")
        return self._root.generate_dot(fill_colors, max_nodes)
