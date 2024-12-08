"""Main script for running supervised learning experiments on Palmer Penguins dataset.

This script runs Decision Tree and Perceptron models on the Palmer Penguins dataset.
The script trains and tests the models using hyperparameter search and saves metrics 
from hyperparameter searches and model performance to output files.

The script can be run with the following command line arguments:
    --all: Run all models
    --dtree1: Run Decision Tree Model 1
    --dtree2: Run Decision Tree Model 2
    --dtree3: Run Decision Tree Model 3
    --dtrees: Run all DT models
    --perceptron1: Run Perceptron 1
    --perceptron2: Run Perceptron 2
    --perceptron-ova: Run Perceptron OVA
    --perceptrons: Run all Perceptron models
    --feature-plot: Generate Feature Plot
    --supress-warn: Suppress UserWarnings
    --no-search: Skip hyperparameter search (use params from prefs).

The script will run the selected models and save the results to output files.
Hyperparameter search is resourced intensive and can be skipped with the --no-search flag.
Estimated run time is about 1 min with hyperparameter search, and under 10 sec without.

Example:
    $ python3 supervised_learning.py --supress-warn 
    $ python3 supervised_learning.py --no-search

Output in the output/ directory of the project:
    - Feature plot of the Palmer Penguins dataset (SVG)
    - Hyperparameter search plots for Decision Tree and Perceptron models (SVG)
    - Confusion matrix plots for final models (SVG)
    - Decision boundary plots for 2-feature Perceptron models (SVG)
    - Decision Tree graphs (SVG)
    - Decision Tree and Perceptron model stats (JSON)
A full run will generate 38 SVG and 6 JSON files in the output/ directory.

During the run, the script will print status updates and time lapsed for each model.
"""

if __name__ != '__main__':
    raise ImportError("This module should not be imported, only run as script.")

import numpy as np
from numpy.typing import NDArray
import graphviz
import matplotlib.pyplot as plt
from typing import Literal, Any, Optional, Union, Iterable
import json
from datetime import datetime
import argparse
import warnings

import data_tools.color_map as dtcm
import data_tools.plot as dtplt
import data_tools.measure as dtm
import data_tools.prepare as dtprep
from data_tools.data_splitter import RandomSplit, KFold
from data_tools.hyperparam_search import RandomSearch
from prefs import (
    PENGUIN_DATA_PATH,
    OUTPUT_DIR,
    PalmerDatasetNames as PDN,
    HyperparamSearchProps as HSP,
    Hyperparams as HYP
)
from models.ml_model import MLModel
from models.decicion_tree import DecisionTree, DTHyperparams
from models.perceptron import Perceptron, PerceptronHyperparams, \
    scatter_plot_decision_boundaries
from models.perceptron_ova import PerceptronOVAClassifier as PerceptronOVA


def load_palmer_penguins(
        feature_plot=False,
) -> tuple[NDArray, NDArray, list, list]:
    """Load the Palmer Penguins dataset.
    
    - Load the Palmer Penguins dataset from a CSV file to NDArray.
    - Select numeric features and target column.
    - Normalize the data.

    Parameters
    ----------
    feature_plot : bool, optional
        Generate feature plot, by default False

    Returns
    -------
    X: NDArray
        Feature matrix
    y: NDArray
        Target vector
    feat_names: list
        Feature names
    feat_names_no_unit: list
        Feature names without units
    y_labels: list
        Target labels
    """
    selected_cols = PDN.NUMERIC_FEATURES + [PDN.Y_COL_NAME]
    headers, data = dtprep.load_csv_numpy(
        path=PENGUIN_DATA_PATH,
        select_columns=selected_cols
    )

    (X_not_norm,
     y,
     feat_names,
     y_labels) = dtprep.prep_data_xy_matrix(
        data,
        headers,
        y_col=PDN.Y_COL_NAME,
        remove_na=True,
        normalize=False
    )

    if feature_plot:
        fig_feature = plt.figure()
        dtplt.feature_plot(X_not_norm, y, feat_names, y_labels,
                        show_plot=False, fig=fig_feature)
        fig_feature.savefig(
            OUTPUT_DIR / f'palmer_penguins_feature_plot.svg')
        plt.close(fig_feature)

    X = dtprep.normalize_data(X_not_norm)
    feat_names_no_unit = [f[:f.rindex('_')] for f in feat_names]
    return (X,
            y,
            feat_names,
            feat_names_no_unit,
            y_labels)

def select_subset(
        selected_features: list[str],
        y_label_group1: str
) -> tuple[NDArray, NDArray, list[str], list[str]]:
    """Select a subset of the Palmer Penguins dataset.
    
    - Select a subset of the loaded dataset based on selected_features.
    - Remap the target vector to binary classes.

    Parameters
    ----------
    selected_features : list
        List of feature names to select
    y_label_group1 : str
        Target label to be remapped to 1

    Returns
    -------
    X_selected: NDArray
        Feature matrix with selected features
    y_subset: NDArray
        Target vector with binary classes
    feat_names_selected: list
        Selected feature names
    y_lbls_subset: list
        Target string labels for binary classes
    """
    cidx_selected = [
        feature_names_all.index(n)
        for n in selected_features
    ]
    X_selected = X_all[:, cidx_selected]
    feat_names_selected = [
        feature_names_no_unit[i]
        for i in cidx_selected
    ]
    y_group1_int = y_labels_all.index(y_label_group1)
    y_subset = dtprep.binary_remap_vector(y_all, y_group1_int)
    y_lbls_subset = ['Other', y_label_group1]
    return (
        X_selected,
        y_subset,
        feat_names_selected,
        y_lbls_subset
    )


def perceptron_train_test(
        X: NDArray,
        y: NDArray,
        y_lbls: NDArray,
        search_params: list[dict],
        feat_names: Optional[NDArray] = None,
        use_hyperparams: Optional[dict] = None
) -> None:
    """Train and test a Perceptron model on the Palmer Penguins dataset.
    
    - Perform hyperparameter search if use_hyperparams is None.
      - If hyperparameter search is performed, save search results and
        set hyperparameters for the model to the best found.
    - Train a Perceptron model on the dataset.
    - Test the model and save performance metrics.
    - Save confusion matrix and decision boundary plots.

    Parameters
    ----------
    X : NDArray
        Feature matrix
    y : NDArray
        Target vector
    y_lbls : NDArray
        Target labels
    search_params : list[dict]
        List of hyperparameter search parameters
    feat_names : NDArray, optional
        Feature names, by default None
    use_hyperparams : dict, optional
        Hyperparameters to use for the model.
        If None, perform hyperparameter search.
    """
    if use_hyperparams is None:
        for h_search_kwargs in search_params:
            use_hyperparams = perceptron_param_search(
                X, y, y_lbls,
                **h_search_kwargs
            )
    hyperparams = PerceptronHyperparams(**use_hyperparams)

    ova = search_params[0].get('ova', False)

    # Train perceptron
    splitter = RandomSplit(train_size=0.8, n_splits=1)
    X_train, X_test, y_train, y_test = splitter.split_Xy_data(X, y)[0]
    n_features = X.shape[1]
    if ova:
        perceptron = PerceptronOVA(n_features, hyperparameters=hyperparams)
    else:
        perceptron = Perceptron(n_features, hyperparams)
    perceptron.fit(X_train, y_train)
    y_test_pred = perceptron.predict(X_test)
    b_score = dtm.prediction_score(y_test, y_test_pred, 'b-score', True)

    # Set up path for output files
    p_id = search_params[0]['id']
    path = str(OUTPUT_DIR / f'perceptron{p_id}_stats')

    # Plot confusion matrix
    y_pred = perceptron.predict(X_test)
    cm = dtm.confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    dtplt.plot_confusion_matrix(cm, y_lbls, ax, show_plot=False)
    fig.savefig(f'{path}_cm.svg')
    plt.close(fig)

    if not ova:
        # Plot decision boundary
        db_slope_intercept = perceptron.decision_boundary_slope_intercept()
        gr1_name = y_lbls[1]
        gr1_orig_yval = y_labels_all.index(gr1_name)
        path_db = f'{path}_db.svg'
        scatter_plot_decision_boundaries(
            X, y_all, db_slope_intercept, y_labels_all, [gr1_orig_yval],
            feat_names, b_score, show_plot=False, save_as=path_db
        )

        # Plot decision boundaries for multiple perceptrons with same parameters
        perceptrons = [Perceptron(n_features, hyperparams) for _ in range(50)]
        for p in perceptrons:
            p.fit(X_train, y_train)
        scores = [
            dtm.prediction_score(y_test, p.predict(X_test), 'b-score', True)
            for p in perceptrons
        ]
        dbs = [p.decision_boundary_slope_intercept() for p in perceptrons]
        path_db_multi = f'{path}_db_multi.svg'
        scatter_plot_decision_boundaries(
            X, y_all, dbs, y_labels_all, [gr1_orig_yval], feat_names,
            scores, show_plot=False, save_as=path_db_multi
        )

    # Save stats
    save_stats(perceptron, X, y, y_lbls, path)


def perceptron_param_search(
        X: NDArray, 
        y: NDArray,
        y_labels: NDArray,
        **search_kwargs: Any
) -> dict:
    """Perform hyperparameter search for a Perceptron model.
    
    - Perform hyperparameter search for a Perceptron model.
    - Save search results and performance metrics to output files.
    
    Parameters
    ----------
    X : NDArray
        Feature matrix
    y : NDArray
        Target vector
    y_labels : NDArray
        Target labels
    **search_kwargs : Any
        Hyperparameter search parameters
    """
    n_features = X.shape[1]
    perceptron_init_args = {
        'n_features': n_features,
    }
    param_ranges = {
        k: v for k, v in search_kwargs.items() 
        if k in ['max_epochs', 'learning_rate', 'accuracy_goal']
    }
    ova = search_kwargs.get('ova', False)
    perceptron_cls = PerceptronOVA if ova else Perceptron
    
    k_fold = KFold(n_splits=5)
    hyperparam_rand_search = RandomSearch(
        model=perceptron_cls,
        param_distributions=param_ranges,
        splitter=k_fold,
        n_iter=search_kwargs['n_iter'],
        scoring='b-score',
        model_init_kwargs=perceptron_init_args
    )
    best_params = hyperparam_rand_search.fit(X, y)
    fig_score = plt.figure()
    fig_cm = plt.figure()
    hyperparam_rand_search.plot_score_cm(
        y_labels, 
        fig_score=fig_score,
        fig_cm=fig_cm,
        show_plot=False,
        title_confusion_matrices='',
        title_hyperparam_scores=''
    )
    perceptron_id = search_kwargs['id']
    search_num = search_kwargs['search_num']
    fname_prefix = f'perceptron{perceptron_id}_search{search_num}'
    fig_score.savefig(
        OUTPUT_DIR / f'{fname_prefix}_score.svg')
    fig_cm.savefig(
        OUTPUT_DIR / f'{fname_prefix}_cm.svg')
    plt.close(fig_score)
    plt.close(fig_cm)
    return best_params


def dtree_train_test(
        X: NDArray,
        y: NDArray,
        y_labels: list[str],
        feat_names: list[str],
        search_params: list[dict],
        use_hyperparams: Optional[dict] = None
) -> None:
    # Optional hyperparameter search
    if use_hyperparams is None:
        for h_search_kwargs in search_params:
            use_hyperparams = dtree_param_search(
                X, y, y_labels,
                **h_search_kwargs
            )
    
    hyperparams = DTHyperparams(**use_hyperparams)

    # Init and train tree, set base path for output files
    splitter = RandomSplit(train_size=0.8, n_splits=1)
    X_train, X_test, y_train, y_test = splitter.split_Xy_data(X, y)[0]
    dtree = DecisionTree(X_train, y_train, hyperparams, feat_names, y_labels)
    tree_num = search_params[0]['id']
    path = str(OUTPUT_DIR / f'dtree{tree_num}_stats')

    # Run tests and save output files with and without post pruning
    dtree_stats(y_labels, X_test, y_test, dtree, path)
    dtree.post_prune()
    dtree_stats(y_labels, X_test, y_test, dtree, path + '_pruned')

def dtree_stats(y_labels, X_test, y_test, dtree, path):
    # Calculate stats
    y_test_pred = dtree.predict(X_test)
    cm_test = dtm.confusion_matrix(y_test, y_test_pred)
    b_score_test = dtm.prediction_score(y_test, y_test_pred, 'b-score')

    # Set up path for output files
    # Save plots
    fig, ax1 = plt.subplots(figsize=(6, 5))
    dtplt.plot_confusion_matrix(cm_test, y_labels, ax1, show_plot=False)
    ax1.set_title(f'Test data (b-score: {b_score_test:.2f})')
    fig.savefig(f'{path}_cm.svg')
    plt.close(fig)

    # Save decision tree graph
    dot_str = dtree.generate_dot()
    gviz = graphviz.Source(dot_str)
    gviz.render(f'{path}_tree', format='svg')

    # Save stats
    save_stats(dtree, X_test, y_test, y_labels, path)


def dtree_param_search(
        X: NDArray, 
        y: NDArray, 
        y_labels: list[str], 
        **search_kwargs: Any
) -> dict:
    param_ranges = {
        k: v for k, v in search_kwargs.items() 
        if k in ['max_depth', 'min_samples_split', 'min_samples_leaf']
    }

    k_fold = KFold(n_splits=5)
    rnd_search = RandomSearch(
        model=DecisionTree,
        param_distributions=param_ranges,
        splitter=k_fold,
        n_iter=search_kwargs['n_iter'],
        scoring='b-score'
    )
    best_params = rnd_search.fit(X, y)
    fig_score, fig_cm = plt.figure(), plt.figure()
    rnd_search.plot_score_cm(
        y_labels, 
        fig_score=fig_score,
        fig_cm=fig_cm,
        show_plot=False
    )
    dtree_id = search_kwargs['id']
    search_num = search_kwargs['search_num']
    path = OUTPUT_DIR / f'dtree{dtree_id}_search{search_num}'
    fig_score.savefig(f'{path}_score.svg')
    fig_cm.savefig(f'{path}_cm.svg')
    plt.close(fig_cm)
    plt.close(fig_score)
    return best_params


def save_stats(
        trained_model: MLModel,
        X: NDArray,
        y: NDArray,
        y_labels: list[str],
        save_as: str
) -> None:
    y_pred = trained_model.predict(X)
    cm = dtm.confusion_matrix(y, y_pred)
    precision, recall = dtm.precision_recall(
        y, y_pred,
        average=None,
        return_vals='both'
    )
    w_mean = dtm.weighted_mean_accuracy(y, y_pred)
    precision_dict = {lbl: float(p) for lbl, p in zip(y_labels, precision)}
    recall_dict = {lbl: float(r) for lbl, r in zip(y_labels, recall)}
    precision_dict['mean'] = float(np.mean(precision))
    recall_dict['mean'] = float(np.mean(recall))
    b_score = dtm.prediction_score(y, y_pred, 'b-score', mean=False)
    f_score = dtm.prediction_score(y, y_pred, 'f-score', mean=False)
    b_score_dict = {lbl: float(b) for lbl, b in zip(y_labels, b_score)}
    f_score_dict = {lbl: float(f) for lbl, f in zip(y_labels, f_score)}
    b_score_dict['mean'] = float(np.mean(b_score))
    f_score_dict['mean'] = float(np.mean(f_score))
    if isinstance(trained_model, MLModel):
        model_props = trained_model.get_model_props(include_hyperparams=True)
    else:
        model_props = ''
    stats = {
        'model': str(trained_model.__class__.__name__),
        'model_props': model_props,
        'confusion_matrix': [[int(i) for i in row] for row in cm],
        'precision': precision_dict,
        'recall': recall_dict,
        'b_score': b_score_dict,
        'f_score': f_score_dict,
        'weighted_mean_accuracy': w_mean,
    }

    with open(save_as + '.json', 'w') as f:
        json.dump(stats, f, indent=4)


def get_time_lapsed_str(time_start, time_end=None) -> str:
    if time_end is None:
        time_end = datetime.now()
    elapsed_time = time_end - time_start
    minutes, seconds = divmod(elapsed_time.total_seconds(), 60)
    return f"{minutes:02g} min, {seconds:02.2f} sec"


def status_update(
        msg: str, 
        t_start: Optional[datetime] = None
) -> None:
    if silent:
        return
    time_lapsed = ''
    if not t_start:
        print(msg)
        return datetime.now()
    time_end = datetime.now()
    time_lapsed = f" ({get_time_lapsed_str(t_start, time_end)})"
    print(f"{msg}{time_lapsed}")


def parse_args():
    descr = "Run training and testing for supervised learning "
    descr += "algorithms on ML models and the Palmer Penguins dataset."
    parser = argparse.ArgumentParser(
        description=descr
    )
    flag_props = [
        ('--all', "Run all models"),
        ('--dtree1', "Run Decision Tree Model 1"),
        ('--dtree2', "Run Decision Tree Model 2"),
        ('--dtree3', "Run Decision Tree Model 3"),
        ('--dtrees', "Run all DT models"),
        ('--perceptron1', "Run Perceptron 1"),
        ('--perceptron2', "Run Perceptron 2"),
        ('--perceptron-ova', "Run Perceptron OVA"),
        ('--perceptrons', "Run all Perceptron models"),
        ('--feature-plot', "Generate Feature Plot"),
        ('--supress-warn', "Suppress UserWarnings"),
        ('--no-search', "Skip hyperparameter search (use params from prefs).")
    ]
    for flag, desc in flag_props:
        parser.add_argument(flag, action='store_true', help=desc)
    args = parser.parse_args()
    args_dict = vars(args)

    if args_dict.pop('supress_warn', False):
        warnings.filterwarnings('ignore', category=UserWarning)

    dtrees = ['dtree1', 'dtree2', 'dtree3']
    perceptrons = ['perceptron1', 'perceptron2', 'perceptron_ova']
    if args_dict.pop('dtrees', False):
        for dtree in dtrees:
            args_dict[dtree] = True
    if args_dict.pop('perceptrons', False):
        for perceptron in perceptrons:
            args_dict[perceptron] = True
    flags = set(k for k, v in args_dict.items() if v)
    if not flags.intersection(dtrees + perceptrons):
        flags.add('all')
    if 'all' in flags:
        flags.update(dtrees, perceptrons, {'feature_plot'})
    return flags


# Parse arguments
run_args = parse_args()
silent = 'silent' in run_args
no_search = 'no_search' in run_args
feat_plot = 'feature_plot' in run_args
load_subset1 = 'perceptron1' in run_args or 'dtree1' in run_args
load_subset2 = 'perceptron2' in run_args or 'dtree2' in run_args

#dbug
#no_search = True

time_start = status_update("\n# Running supervised learning models on Palmer Penguins dataset. #\n")

# Load data
(X_all,
 y_all,
 feature_names_all,
 feature_names_no_unit,
 y_labels_all) = load_palmer_penguins(feat_plot)
status_update('- Original Palmer Penguin dataset loaded.')

# Load subsets for specific models, if selected
if load_subset1:
    selected_features = [PDN.F_BILL_DEPTH, PDN.F_FLIPPER_LENGTH]
    y_lbl_group1 = 'Gentoo'
    (X_bill_d_flip_l,
     y_gentoo,
     feat_names_bill_d_flip_l,
     y_lbls_gentoo) = select_subset(selected_features, y_lbl_group1)
    status_update('- Subset for Gentoo penguins loaded.')
if load_subset2:
    selected_features = [PDN.F_BILL_LENGTH, PDN.F_BILL_DEPTH]
    y_lbl_group1 = 'Chinstrap'
    (X_bill_l_d,
     y_chinstrap,
     feat_names_bill_l_d,
     y_lbls_chinstrap) = select_subset(selected_features, y_lbl_group1)
    status_update('- Subset for Chinstrap penguins loaded.')

# Train and test selected ML models
if no_search:
    status_update("\n** Skipping hyperparameter search **" +
                  "\n(loading hyperparams from prefs.py)")

if 'perceptron1' in run_args:
    time_sub = status_update("\n## Perceptron 1 ##")
    hyperparams = HYP.PERCEPTRON_1 if no_search else None
    perceptron_train_test(
        X_bill_d_flip_l, y_gentoo, y_lbls_gentoo,
        feat_names=feat_names_bill_d_flip_l,
        search_params=HSP.PERCEPTRON_1,
        use_hyperparams=HYP.PERCEPTRON_1 if no_search else None
    )
    status_update("Perceptron 1 done", time_sub)

if 'perceptron2' in run_args:
    time_sub = status_update("\n## Perceptron 2 ##")
    perceptron_train_test(
        X_bill_l_d, y_chinstrap, y_lbls_chinstrap,
        feat_names=feat_names_bill_l_d,
        search_params=HSP.PERCEPTRON_2,
        use_hyperparams=HYP.PERCEPTRON_2 if no_search else None
    )
    status_update("Perceptron 2 done", time_sub)

if 'perceptron_ova' in run_args:
    time_sub = status_update("\n## Perceptron OVA ##")
    perceptron_train_test(
        X_all, y_all, y_labels_all,
        search_params=HSP.PERCEPTRON_OVA,
        use_hyperparams=HYP.PERCEPTRON_OVA if no_search else None
    )
    status_update("Perceptron OVA done", time_sub)

if 'dtree1' in run_args:
    time_sub = status_update("\n## Decision Tree 1 ##")
    dtree_train_test(
        X_bill_d_flip_l, y_gentoo,
        y_lbls_gentoo, feat_names_bill_d_flip_l,
        search_params=HSP.DTREE_1,
        use_hyperparams=HYP.DTREE_1 if no_search else None
    )
    status_update("DT1 done", time_sub)

if 'dtree2' in run_args:
    time_sub = status_update("\n## Decision Tree 2 ##")
    dtree_train_test(
        X_bill_l_d, y_chinstrap,
        y_lbls_chinstrap, feat_names_bill_l_d,
        search_params=HSP.DTREE_2,
        use_hyperparams=HYP.DTREE_2 if no_search else None
    )
    status_update("DT2 done", time_sub)

if 'dtree3' in run_args:
    time_sub = status_update("\n## Decision Tree 3 ##")
    dtree_train_test(
        X_all, y_all, y_labels_all, feature_names_no_unit,
        search_params=HSP.DTREE_3,
        use_hyperparams=HYP.DTREE_3 if no_search else None
    )
    # To show pruning effect, run DT3 again with no pruning
    dtree_train_test(
        X_all, y_all, y_labels_all, feature_names_no_unit,
        search_params=[{'id': '3-no-prune'}],  # For output file name
        use_hyperparams={}  # Use default hyperparams
    )
    status_update("DT3 done", time_sub)

status_update("- Total time:", time_start)