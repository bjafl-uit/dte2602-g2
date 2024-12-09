"""Module for storing constants and preferences for supervised_learning.py.

Contains constants for the Palmer penguins dataset, hyperparameters for the
perceptron and decision tree models, and hyperparameter search properties.
The constants are used in supervised_learning.py to load the dataset, set
hyperparameters, and perform hyperparameter searches.

Attributes
----------
ASSETS_DIR : Path
    The directory containing the dataset and output files.
PENGUIN_DATA_PATH : Path
    The path to the Palmer penguins dataset.
OUTPUT_DIR : Path
    The directory to save output files.
PalmerDatasetNames : class
    Constants for the Palmer penguins dataset.
Hyperparams : class
    Hyperparameters for the perceptron and decision tree models.
HyperparamSearchProps : class
    Hyperparameter search properties for the perceptron and decision 
    tree models.

Notes
-----
- The Palmer penguins dataset is available at:
    https://allisonhorst.github.io/palmerpenguins/
- The classes in this module are only wrappers for constants and are not
    intended to be instantiated.
"""

from pathlib import Path
import numpy as np

# File paths
ASSETS_DIR = Path(__file__).parent / 'assets'
PENGUIN_DATA_PATH = ASSETS_DIR / 'palmer_penguins.csv'
OUTPUT_DIR = ASSETS_DIR.parent / 'output'



class PalmerDatasetNames:
    """Constants for the Palmer penguins dataset.

    Attributes
    ----------
    Y_COL_NAME : str
        The name of the target column.
    Y_(...) : str
        The possible target values.
    F_(...) : str
        Feature column names
    """

    Y_COL_NAME = 'species'
    Y_ADELIE = 'Adelie'
    Y_CHINSTRAP = 'Chinstrap'
    Y_GENTOO = 'Gentoo'
    F_BILL_LENGTH = 'bill_length_mm'
    F_BILL_DEPTH = 'bill_depth_mm'
    F_FLIPPER_LENGTH = 'flipper_length_mm'
    F_BODY_MASS = 'body_mass_g'

    NUMERIC_FEATURES = [
        F_BILL_DEPTH,
        F_BILL_LENGTH,
        F_FLIPPER_LENGTH,
        F_BODY_MASS
    ]

class Hyperparams:
    PERCEPTRON_1 = {
        'max_epochs': 300,
        'learning_rate': 0.1,
        'accuracy_goal': 1.0
    }
    PERCEPTRON_2 = {
        'max_epochs': 300,
        'learning_rate': 0.095,
        'accuracy_goal': 1.0
    }
    PERCEPTRON_OVA = {
        'max_epochs': 300,
        'learning_rate': 0.095,
        'accuracy_goal': 1.0
    }
    DTREE_1 = {
        'max_depth': 5,
        'min_samples_leaf': 5,
        'min_samples_split': 10
    }
    DTREE_2 = {
        'max_depth': 8,
        'min_samples_leaf': 50,
        'min_samples_split': 100,
    }
    DTREE_3 = {
        'max_depth': 10,
        'min_samples_leaf': 8,
        'min_samples_split': 16
    }
    DTREE_3_NO_PRUNE = {
        'max_depth': None,
        'min_samples_leaf': 1,
        'min_samples_split': 2,
    }


class HyperparamSearchProps:
    PERCEPTRON_1 = [
        {
            'id': 1,
            'search_num': 1,
            'n_iter': 100,
            'max_epochs': np.arange(100, 601, 100),
            'learning_rate': np.linspace(0.01, 0.15, 100),
            'accuracy_goal': np.linspace(0.9, 1.0, 11)
        },
        {
            'id': 1,
            'search_num': 2,
            'n_iter': 50,
            'max_epochs': np.arange(100, 401, 50),
            'learning_rate': np.linspace(0.05, 0.15, 30),
            'accuracy_goal': [1.0]
        }
    ]
    PERCEPTRON_2 = [
        {
            'id': 2,
            'search_num': 1,
            'n_iter': 100,
            'max_epochs': np.arange(100, 601, 50),
            'learning_rate': np.linspace(0.01, 0.15, 30),
            'accuracy_goal': np.linspace(0.9, 1.0, 5),
        },
        {
            'id': 2,
            'search_num': 2,
            'n_iter': 50,
            'max_epochs': [300, 400],
            'learning_rate': np.linspace(0.04, 0.15, 30),
            'accuracy_goal': [1.0],
        }
    ]
    PERCEPTRON_OVA = [
        {
            'id': '_ova',
            'ova': True,
            'search_num': 1,
            'n_iter': 100,
            'max_epochs': np.arange(100, 601, 50),
            'learning_rate': np.linspace(0.01, 0.15, 30),
            'accuracy_goal': np.linspace(0.9, 1.0, 5),
        },
        {
            'id': '_ova',
            'ova': True,
            'search_num': 2,
            'n_iter': 50,
            'max_epochs': [300, 400],
            'learning_rate': np.linspace(0.05, 0.15, 20),
            'accuracy_goal': [1.0],
        }
    ]
    PERCEPTRON_OVA_EXT = [
        {
            'id': '_ova_ext',
            'ova': True,
            'search_num': 1,
            'n_iter': 50,
            'max_epochs': np.arange(300, 501, 100),
            'learning_rate': np.linspace(0.01, 0.15, 20),
            'accuracy_goal': np.linspace(0.3, 1.0, 10),
        }
    ]
    DTREE_1 = [
        {
            'id': 1,
            'search_num': 1,
            'n_iter': 100,
            'max_depth': (1, 10),
            'min_samples_split': (2, 30),
            'min_samples_leaf': (1, 30),
        },
        {
            'id': 1,
            'search_num': 2,
            'n_iter': 50,
            'max_depth': [5],
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10)
        }
    ]
    DTREE_2 = [
        {
            'id': 2,
            'search_num': 1,
            'n_iter': 200,
            'max_depth': (2, 15),
            'min_samples_split': (2, 150),
            'min_samples_leaf': (1, 75)
        },
        {
            'id': 2,
            'search_num': 2,
            'n_iter': 100,
            'max_depth': [8],
            'min_samples_split': (50, 100),
            'min_samples_leaf': (25, 50)
        }
        
    ]
    DTREE_3 = [
        {
            'id': 3,
            'search_num': 1,
            'n_iter': 100,
            'max_depth': (3, 30),
            'min_samples_split': (2, 50),
            'min_samples_leaf': (1, 50),
        },
        {
            'id': 3,
            'search_num': 2,
            'n_iter': 50,
            'max_depth': [10],
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
        }
    ]
