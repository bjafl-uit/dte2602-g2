"""Functions for preparing data for machine learning.

This module contains functions for loading and preparing data for machine
learning tasks. The functions provide functionality for loading CSV files
into NumPy arrays and pre process data matrices for machine learning.

Methods
-------
load_csv_numpy
    Load a CSV file into a NumPy array.
prep_data_xy_matrix
    Prepare data matrix for machine learning.
remove_na_rows
    Remove rows with NA values from a NumPy array.
convert_string_columns
    Convert string columns to numeric columns.
convert_class_vector_to_int
    Convert class vector to integer vector.
normalize_data
    Normalize data in array to z-scores.
binary_remap_vector
    Remap values in a vector to binary values.
join_words
    Join sequence of words into a sentence.
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Optional, Literal


def load_csv_numpy(
        path: str,
        delimiter: str = ',',
        select_columns: Optional[list[int | str]] = None,
        convert_to: Literal[None, 'float', 'int'] = 'float',
        ignore_conversion_error: bool = True
) -> tuple[NDArray, list[str]]:
    """Load a CSV file into a NumPy array.

    Parameters
    ----------
    path : str
        The path to the CSV file.
    delimiter : str
        The delimiter used in the CSV file.
    select_columns : list[int], optional
        The columns to select from the CSV file. If None, all columns are
        selected.
    convert_to : {None, 'float', 'int'}, default='float'
        The type to convert the data to. If None, no conversion is done.
    ignore_conversion_error : bool
        Whether to ignore errors when converting the data.

    Returns
    -------
    headers : list
        The headers of the CSV file.
    data : np.ndarray
        The data of the CSV file.

    Raises
    ------
    ValueError
        If the data could not be converted to the specified type, and
        ignore_conversion_error is False
    """
    data = np.loadtxt(path, delimiter=delimiter, dtype=str)
    headers = [str(h) for h in data[0]]
    data = data[1:]
    if select_columns is not None:
        cols = np.array(select_columns)
        if not np.issubdtype(cols.dtype, np.integer):
            cols = np.array([headers.index(col) for col in select_columns])
        data = data[:, cols]
        headers = [headers[i] for i in cols]
    if convert_to is not None:
        type = int if convert_to == 'int' else float
        try:
            data = data.astype(type)
        except ValueError:
            if not ignore_conversion_error:
                raise ValueError(f'Could not convert data to {convert_to}.')
    return headers, data

def prep_data_xy_matrix(
        data: ArrayLike, 
        headers: list[str],
        y_col: int | str,
        remove_na: bool = True,
        normalize: bool = False,
        y_lbl_to_int: bool = True,
) -> tuple[NDArray, NDArray, list[str], list[str] | None]:
    """Prepare data matrix for machine learning.

    Data pre-processing for machine learning tasks. The function splits
    the target column from the feature matrix, converts feature falues 
    to numbers and further processes the data as specified:
        - Remove rows with NA values
        - Normalize the data
        - Convert class labels in target column to integers

    Parameters
    ----------
    data : np.ndarray
        The data to prepare.
    headers : list
        The headers of the data.
    y_col : int or str
        The column index or name of the target variable.
    remove_na : bool, default=True
        Whether to remove rows with NA values.
    normalize : bool, default=False
        Whether to normalize the data.
    y_lbl_to_int : bool, default=True
        Whether to convert class string labels to integers.

    Returns
    -------
    X : np.ndarray
        The feature matrix.
    y : np.ndarray
        The target vector.
    feature_names : list
        The names of the features.
    y_labels : list or None
        The labels of the target variable, if it is a class variable.
    """
    if isinstance(y_col, str) and headers is not None:
        y_col = headers.index(y_col)
    elif not isinstance(y_col, int):
        raise ValueError("y_col must be an integer or a string")
    
    if remove_na:
        data = remove_na_rows(data)
    y = data[:, y_col]
    X = np.delete(data, y_col, axis=1)
    feature_names = headers[:y_col] + headers[y_col+1:]
    
    X, _ = convert_string_columns(X)
    if normalize:
        X = normalize_data(X)
    
    
    y_labels = None
    try:
        y = y.astype(float)
    except ValueError:
        if y_lbl_to_int:
            y, y_labels = convert_class_vector_to_int(y)
    
    return X, y, feature_names, y_labels

def remove_na_rows(
        data: NDArray,
        na_values: list[str] = ['NA', 'na', '.', '', 'nan', 'NaN', np.nan]
) -> NDArray:
    """Remove NA values from a NumPy array.

    Remove observations (rows) with NA values from a NumPy array.

    Parameters
    ----------
    data : np.ndarray
        The data to remove NA values from.
    na_values : list
        The values to consider as NA.

    Returns
    -------
    data : np.ndarray
        The data with NA values removed.
    """
    return data[~np.any(np.isin(data, na_values), axis=1)]


def convert_string_columns(
        data: NDArray
) -> tuple[NDArray, dict[int, list[str]]]:
    """Convert string columns to numeric columns.

    Checks each column in the data array. If the column contains strings,
    the unique values in the column are converted to integers. The data
    array is then updated with the new integer values. A dictionary is
    returned that maps the column number to the unique values in the column
    that were converted.

    Parameters
    ----------
    data : np.ndarray
        The data to convert.

    Returns
    -------
    data : np.ndarray
        The data with string columns converted to numeric columns.
    converted_cols : dict[int, list[str]]
        A dictionary mapping the column number to lists of the unique values
        in each column that were converted. The indices of the list items
        correspond to the new numeric values used in the converted data array.
    """
    non_numeric_cols = []
    for c_nr in range(data.shape[1]):
        try:
            _ = data[:, c_nr].astype(float)
        except ValueError:
            non_numeric_cols.append(c_nr)

    converted_cols = {}
    new_data = np.zeros_like(data, dtype=float)
    col_mask = ~np.isin(np.arange(data.shape[1]), non_numeric_cols)
    new_data[:, col_mask] = data[:, col_mask].astype(float)
    for c_nr in non_numeric_cols:
        unique_values = np.unique(new_data[:, c_nr])
        value_map = {value: i for i, value in enumerate(unique_values)}
        new_data[:, c_nr] = np.array([value_map[value]
                                      for value in new_data[:, c_nr]], dtype=int)
        converted_cols[c_nr] = [str(v) for v in unique_values]

    return new_data, converted_cols

def convert_class_vector_to_int(v: NDArray) -> tuple[NDArray, list]:
    """Convert class vector to integer vector.

    Converts a class vector to an integer vector. The unique values in the
    class vector are converted to integers. The class vector is then updated
    with the new integer values. A list of the unique values in the class
    vector is returned as reference.

    Parameters
    ----------
    v : np.ndarray
        The class vector to convert.
  
    Returns
    -------
    v : np.ndarray[int]
        The class vector converted to integers.
    unique_values : list
        The unique values in the class vector. The indices of the list
        correspond to the new integer values used in the converted vector.
    """
    unique_values = np.unique(v)
    value_map = {value: i for i, value in enumerate(unique_values)}

    new_vector = np.array([value_map[value] for value in v], dtype=int)
    return new_vector, [str(v) for v in unique_values]


def normalize_data(data: NDArray) -> NDArray:
    """Normalize data in array to z-scores.

    Normalize the data in an array to z-scores. The data is normalized
    column-wise.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        The data to normalize.

    Returns
    -------
    data : np.ndarray, shape (n_samples, n_features)
        The normalized data (z-scores).

    Notes
    -----
    - The z-score is calculated as (f_n - mean(F_n)) / std(F_n), where f_n
        is a feature value, mean(F_n) is the mean of all f_n values and std(F_n)
        is the standard deviation of all f_n values.
    - The data is excpected to be in the shape (n_samples, n_features).
    """
    return (data - data.mean(axis=0)) / data.std(axis=0)


def binary_remap_vector(
        vector: NDArray,
        group1_values: list,
) -> NDArray:
    """Remap values in a vector to binary values.

    Takes a class vector and remaps the values to binary values. The values
    in group1_values are remapped to 1, all other values are remapped to 0.

    Parameters
    ----------
    vector : np.ndarray
        The vector to remap.
    group1_values : list
        The values to remap to 1. All other values are remapped to 0.

    Returns
    -------
    new_vector : np.ndarray[int]
        The vector with remapped values.
    """
    new_vector = np.zeros(vector.shape, dtype=int)
    gr1_mask = np.isin(vector, group1_values)
    new_vector[gr1_mask] = 1
    return new_vector


def join_words(
        lst: ArrayLike,
        last_item_joined_by: str = 'or'
) -> str:
    """Join sequence of words into a sentence.
    
    Join a sequence of words into a sentence. The last word is joined by the
    specified word.

    Example:
    --------
    >>> join_words(['red', 'green', 'blue'], 'and')
    'red, green and blue'
    >>> join_words(['red', 'green', 'blue'])
    'red, green or blue'
    
    Parameters
    ----------
    lst : array-like
        The sequence of words to join.
    last_item_joined_by : str, default='or'
        The word to join the two last words by.
    """
    lst = np.array(lst, dtype=str)
    n = len(lst)
    if n == 0:
        return ''
    if n == 1:
        return lst[0]
    return f"{', '.join(lst[:-1])} {last_item_joined_by} {lst[-1]}"
