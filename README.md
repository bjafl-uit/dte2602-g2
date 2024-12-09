# Graded Assignment 2: Supervised Learning

> Course:  DTE2602
>
> Student: Bjarte FløLode
>
> _Uit (Norges arktiske universitet)_
>
> Fall 2024

Contains my code subission and report for the graded assignment _Supervised Learning_. 

- The report is the [pdf in the project root](dte2602-fa24-graded2-bfl.pdf). 
- The main script that runs the experiments and demonstrates the implemented modules
is contained in `supervised_learning.py` in the project root.
- The file [page count](page%20count.odt) is supplied for easy measuring of page count for the
text content in the report. 

## Repository Structure

- `assets/`
  - `palmer_penguins.csv`: The dataset used for training and testing the models.

- `data_tools/`
  - `color_map.py`: Utilities for color mapping.
  - `data_splitter.py`: Tools for splitting datasets into training and testing sets.
  - `gini.py`: Functions for calculating Gini impurity and impurity reduction.
  - `hyperparam_search.py`: Classes and functions for hyperparameter search.
  - `measure.py`: Functions for measuring model performance.
  - `plot.py`: Functions for plotting model performance and data features.
  - `prepare.py`: Functions for preparing and normalizing data.

- `models/`
  - `decicion_tree_nodes.py`: Classes for decision tree nodes and their statistics.
  - `decicion_tree.py`: Implementation of the Decision Tree model.
  - `ml_model.py`: Abstract base class for machine learning models.
  - `perceptron_ova.py`: Implementation of the Perceptron One-vs-All classifier.
  - `perceptron.py`: Implementation of the Perceptron model.

- `output/`: Directory for storing output files such as model performance metrics and plots.

- `prefs.py`: Configuration file for dataset paths and hyperparameter settings.

- `supervised_learning.py`: Main script for running supervised learning experiments on the Palmer Penguins dataset.

## How to Run

To run the supervised learning experiments, run the file `supervised_learning.py` as a python
script. The script accepths the following command-line arguments:

- `--all`: Run all models
- `--dtree1`: Run Decision Tree Model 1
- `--dtree2`: Run Decision Tree Model 2
- `--dtree3`: Run Decision Tree Model 3
- `--dtrees`: Run all Decision Tree models
- `--perceptron1`: Run Perceptron Model 1
- `--perceptron2`: Run Perceptron Model 2
- `--perceptron-ova`: Run Perceptron One-vs-All model
- `--perceptrons`: Run all Perceptron models
- `--feature-plot`: Generate feature plot
- `--supress-warn`: Suppress UserWarnings
- `--no-search`: Skip hyperparameter search (use parameters from `prefs.py`)

The results will be saved in the `output/` directory.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib


> ### Optional
> - Graphviz 
> 
> _To generate SVG files from DOT graphs of Descicion Trees_

## License

This project is licensed under the MIT License.