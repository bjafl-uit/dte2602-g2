"""Decision tree nodes for decision tree model.

This module contains classes for decision tree nodes, which are used
by DecisionTree to represent the nodes of the decision tree.

Classes
-------
DTNode
    Base class for decision tree nodes
DTBranchNode
    Decision node for decision tree
DTLeafNode
    Leaf node for decision tree

Notes
-----
- DTNode is a base class for decision tree nodes. It should not be
  instantiated directly, but is used as a base class for DTBranchNode
  and DTLeafNode.
"""
from __future__ import annotations
from typing import Union, Optional, TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from abc import ABC, abstractmethod

from data_tools.color_map import convert_color
from data_tools.gini import gini_impurity_from_freq

if TYPE_CHECKING:
    from .decicion_tree import DecisionTree

@dataclass
class DTNodeStats:
    """Node statistics for decision tree nodes.

    Attributes
    ----------
    gini: float
        Gini impurity of node
    n_samples: int
        Number of samples in node
    n_samples_per_class: list[int]
        Number of samples per class in node
    """

    gini: float
    n_samples: int
    n_samples_per_class: list[int]

    def __str__(self) -> str:
        """Return string representation of node statistics."""
        lines = [
            f"gini = {self.gini:.3f}",
            f"samples = {self.n_samples}",
            f"value = {self.n_samples_per_class}"
        ]
        return "\n".join(lines)


class DTNode(ABC):
    """Base class for decision tree nodes.

    Base class for decision tree nodes. Should not be instantiated directly.
    Contains common methods and properties for decision tree nodes.

    Constants
    ----------
    _GVIZ_CHARS_TO_ESCAPE: list[str]
        List of characters to escape in Graphviz labels, 
        used in DOT representation of decision tree.
    """
    _GVIZ_CHARS_TO_ESCAPE = [
        # Escape characters for Graphviz labels
        # https://graphviz.gitlab.io/_pages/doc/info/attrs.html#d:label
        # https://graphviz.gitlab.io/_pages/doc/info/shapes.html#record
        '{', '}', '"', '\\', '<', '>', '|'
    ]

    def __init__(
            self,
            value: int,
            tree: Optional[DecisionTree] = None,
            stats: Optional[DTNodeStats] = None
    ) -> None:
        """Initialize decision tree node.

        Parameters
        ----------
        tree: DecisionTree
            Ref to parent tree. Used to access metadata for
            string representation of tree nodes.
        stats: DTNodeStats
            Node statistics

        Notes
        -----
        - Use child classes DecisionTreeBranchNode and DecisionTreeLeafNode
          to create decision and leaf nodes, respectively. This class is
          only used as a base class for these two, and should not be
          instantiated directly.
        """
        self._value = value
        self._tree = tree
        self._stats = stats

    @property
    def tree(self) -> DecisionTree:
        """Return reference to parent tree."""
        return self._tree

    @property
    def stats(self) -> DTNodeStats:
        """Return node statistics."""
        return self._stats

    @property
    def value(self) -> int:
        """Return value of node.

        Returns
        -------
        value: int
            Integer id of class represented by node.

        Notes
        -----
        - For branch nodes, the value is the class label with the
          majority of samples in the node.
        """
        return self._value

    @property
    def value_label(self) -> str:
        """Return label of node value.

        Returns
        -------
        label: str
            String label of class represented by node.

        Raises
        ------
        AttributeError
            If parent tree is not set.
        ValueError
            If class label for the class id is not found in parent tree.

        Notes
        -----
        - For branch nodes, the value is the class label with the
          majority of samples in the node.
        - The class label is retrieved from the parent tree metadata.
        """
        try:
            return self._tree.get_class_label(self._value)
        except AttributeError:
            raise AttributeError("Parent tree not set")
        except ValueError:
            raise ValueError(f"Class label not found for value {self._value}")

    def __str__(self) -> str:
        """Return string representation of decision tree node."""
        lines = []
        if self._stats is not None:
            lines.append(str(self._stats))
        try:
            label = self.value_label
        except ValueError or AttributeError:
            label = self._value
        lines.append(f"class = {label}")
        return "\n".join(lines)

    @abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for input data.

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray
            NumPy array of class labels, shape (n_samples,)

        Notes
        -----
        This method is implemented in child classes DecisionTreeBranchNode
        and DecisionTreeLeafNode. It should not be called on an instance
        of DecicionTreeNode.
        """
        pass
    
    def post_prune(self) -> None:
        """Prune the decision tree.

        Notes
        -----
        - This method is dependent on the implementation of the _post_prune()
            method in the child classes.
        """
        _, n_pruned = self._post_prune()
        return n_pruned
    
    @abstractmethod
    def _post_prune(self) -> None:
        """Prune the decision tree.

        Notes
        -----
        This method should be overridden in the child classes.
        """
        pass

    def generate_dot(self, fill_colors: bool = True, max_nodes: int = 1000) -> str:
        """Generate DOT representation of decision tree.

        Returns
        -------
        dot_str: str
            DOT representation of decision tree

        Notes
        -----
        - _dot_generate() should be overridden by branch class to recursively
          generate child nodes.
        - Inspired by scikit-learn's export_graphviz() function:
        https://scikit-learn.org/1.5/modules/generated/sklearn.tree.export_graphviz.html
        """
        if fill_colors and self._tree is None:
            raise ValueError("Parent tree not defined, cannot set fill colors")

        dot_head_lines = ["digraph Tree {"]
        if fill_colors:
            dot_head_lines.append(
                '    node [shape=box, style="filled", ' +
                'color="black", fontname="helvetica"] ;'
            )
        else:
            dot_head_lines.append(
                '    node [shape=box, fontname="helvetica"] ;'
            )
        dot_head_lines.append('    edge [fontname="helvetica"] ;')

        dot_lines = []
        node_counter = iter(range(max_nodes))
        try:
            self._dot_generate(node_counter, dot_lines)
        except StopIteration:
            raise ValueError("Max number of nodes exceeded")

        dot_lines.sort(
            key=lambda l: int(w[0]) if (w:= l.split()) \
                and w[0].isdigit() else l
        )
        dot_lines = dot_head_lines + dot_lines + ["}"]
        return "\n".join(dot_lines)

    def _dot_escape_label(self, label: str) -> str:
        """Escape special characters in graphviz label strings."""
        for char in self._GVIZ_CHARS_TO_ESCAPE:
            label = label.replace(char, fr"\{char}")
        label = label.replace("\n", r"\n")
        return label

    def _dot_generate(
            self,
            node_counter,
            dot_lines: list[str]) -> str:
        """Generate DOT representation of decision tree.

        Parameters
        ----------
        node: DTNode
            Decision tree node
        node_counter: iter
            Iterator for node ids
        dot_lines: list[str]
            List of DOT lines

        Notes
        -----
        - This method should be overridden in to recursively generate the child
          nodes. The base class implementation only generates a single node.
        """
        label = self._dot_escape_label(str(self))
        node_id = next(node_counter)
        color = self._get_distribution_color()
        dot_lines.append(
            f'    {node_id} [label="{label}", fillcolor="{color}"] ;'
        )
        return node_id

    def _get_distribution_color(self) -> str:
        """Return color for node based on node distribution.

        Returns
        -------
        hex_color: str
            Hexadecimal color string

        Notes
        -----
        - The base color is assigned based on most frequent class in the node
          distribution.
        - This color is lightened based on the proportion of samples of this
          class in the node distribution, relative to an even split.
        - Even split gives white color.
        - Colors are assigned based on the tab10 colormap from matplotlib.
        """
        distribution = self._stats.n_samples_per_class
        p = distribution[self.value] / np.sum(distribution)
        lighten_factor = min((1 - p), 0.5) / 0.5  # 0.5 = even split -> white
        color = self._tree.color_map.get_color(self.value, lighten_factor)
        return convert_color(color, 'hex')


class DTBranchNode(DTNode):
    """Decision tree branch node.

    Branch node for decision tree model.
    """
    def __init__(
            self,
            value: int,
            feature_index: int,
            feature_value: float,
            left: DTNode,
            right: DTNode,
            tree: Optional[DecisionTree] = None,
            node_stats: Optional[DTNodeStats] = None
    ) -> None:
        """Initialize decision node

        Parameters
        ----------
        feature_index: int
            Index of X column used in question
        feature_value: float
            Value of feature used in question
        left: DecisionTreeBranchNode or DecisionTreeLeafNode
            Node, root of left subtree
        right: DecisionTreeBranchNode or DecisionTreeLeafNode
            Node, root of right subtree
        tree: DecisionTree
            Ref to parent tree. Used to access metadata for
            string representation of tree nodes.
        node_stats: DTNodeStats
            Node statistics

        Notes
        -----
        - DecisionTreeBranchNode is a subclass of binarytree.Node. This
        has the advantage of inheriting useful methods for general binary
        trees, e.g. visualization through the __str__ method.
        - Each decision node corresponds to a question of the form
        "is feature x <= value y". The features and values are stored as
        attributes "feature_index" and "feature_value".
        - A string representation of the question is saved in the node's
        "value" attribute.
        """
        super().__init__(value, tree, node_stats)
        self._feature_index = feature_index
        self._feature_value = feature_value
        self._left = left
        self._right = right

    @property
    def feature_index(self) -> int:
        """Return index of feature used in question."""
        return self._feature_index

    @property
    def feature_value(self) -> float:
        """Return value of feature used in question."""
        return self._feature_value

    @property
    def left(self) -> DTNode:
        """Return root of left subtree."""
        return self._left

    @property
    def right(self) -> DTNode:
        """Return root of right subtree."""
        return self._right

    def __str__(self) -> str:
        """Return string representation of decision node."""
        try:
            feature_name = self._tree.get_feature_name(self._feature_index)
        except ValueError:
            feature_name = f"f{self._feature_index}"
        feature_str = f"{feature_name} <= {self._feature_value:.3g}"
        return f"{feature_str}\n{super().__str__()}"

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for input data.

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray
            NumPy array of class labels, shape (n_samples,)
        """
        y = np.zeros(X.shape[0], dtype=int)
        left_mask = X[:, self.feature_index] <= self.feature_value
        y[left_mask] = self.left.predict(X[left_mask])
        y[~left_mask] = self.right.predict(X[~left_mask])
        return y
    
    def _post_prune(self) -> None:
        pruned_node_l, n_pruned_l = self.left._post_prune()
        pruned_node_r, n_pruned_r = self.right._post_prune()
        n_pruned = n_pruned_l + n_pruned_r 
        if isinstance(pruned_node_l, DTLeafNode) and isinstance(pruned_node_r, DTLeafNode):
            if pruned_node_l.value == pruned_node_r.value:
                n_samples = pruned_node_l.stats.n_samples + pruned_node_r.stats.n_samples
                n_samples_per_class = [
                    pruned_node_l.stats.n_samples_per_class[i] + pruned_node_r.stats.n_samples_per_class[i]
                    for i in range(len(pruned_node_l.stats.n_samples_per_class))
                ]
                gini = gini_impurity_from_freq(n_samples_per_class)
                stats = DTNodeStats(gini, n_samples, n_samples_per_class)
                return DTLeafNode(pruned_node_l.value, self.tree, stats), n_pruned + 2
        self._left = pruned_node_l
        self._right = pruned_node_r
        return self, n_pruned

    def _dot_generate(
            self,
            node_counter,
            dot_lines: list[str]) -> str:
        """Generate DOT representation of decision tree.

        Parameters
        ----------
        node: DTNode
            Decision tree node
        node_counter: iter
            Iterator for node ids
        dot_lines: list[str]
            List of DOT lines

        Notes
        -----
        - This method should be overridden in to recursively generate the child
          nodes. The base class implementation only generates a single node.
        """
        node_id = super()._dot_generate(node_counter, dot_lines)
        root_arrow_lbls = ['', '']
        if node_id == 0:
            root_arrow_lbls = [
                f'[labeldistance=2.5, labelangle=45, headlabel="True"] ',
                f'[labeldistance=2.5, labelangle=-45, headlabel="False"] ',
            ]
        child_nodes = [self.left, self.right]
        for child_node, root_lbl in zip(child_nodes, root_arrow_lbls):
            child_id = child_node._dot_generate(node_counter, dot_lines)
            dot_lines.append(
                f'    {node_id} -> {child_id} {root_lbl};'
            )
        return node_id


class DTLeafNode(DTNode):
    """Decision tree leaf node.
    
    Leaf node for decision tree model. 
    """
    def __init__(
            self,
            y_value: int,
            tree: Optional[DecisionTree] = None,
            node_stats: Optional[DTNodeStats] = None
    ) -> None:
        """Initialize leaf node.

        Parameters
        ----------
        y_value: int
            Integer id of class represented by leaf.
        """
        super().__init__(y_value, tree, node_stats)

    def predict(self, X: NDArray) -> NDArray:
        """Predict class labels for input data.

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray
            NumPy array of class labels, shape (n_samples,)
        """
        return np.full(X.shape[0], self._value)
    
    def _post_prune(self) -> None:
        return self, 0
