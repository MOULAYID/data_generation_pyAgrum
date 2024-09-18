import pyAgrum as gum
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import statistics
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import register_adapter
from scipy.stats import wasserstein_distance
import math
from sklearn.metrics import mean_squared_error, f1_score
import data_generation as dg

# Register numpy int64 adapter for PostgreSQL
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)


# ____________________________________________________________________________________________________________________________________________________
def load_BN_from_bif(bn_path):
    """
    Load a Bayesian network (BN) from a .bif file.

    Parameters:
    - bn_path: Path to the .bif file.

    Returns:
    - The loaded Bayesian network.
    """
    return gum.loadBN(bn_path)


# ____________________________________________________________________________________________________________________________________________________
def compute_marginal_distribution(dataframe, variable_index):
    """
    Compute the marginal probability distribution for a specific variable in a DataFrame.

    Parameters:
    - dataframe: The DataFrame containing the data.
    - variable_index: The index of the variable for which to compute the marginal distribution.

    Returns:
    - A list representing the ordered marginal probability distribution.
    """
    variable_name = dataframe.columns[variable_index]
    marginal_distribution = dataframe.iloc[:, variable_index].value_counts(normalize=True).sort_index().tolist()
    return marginal_distribution


# ____________________________________________________________________________________________________________________________________________________
def joint_dist_bn_dict(bn):
    """
    Compute the joint probability distribution of a Bayesian network as a dictionary.

    Parameters:
    - bn: The Bayesian network.

    Returns:
    - A dictionary where keys are tuples of variable values and values are probabilities.
    """
    var_list_names = sorted(bn.names())
    jd = gum.Potential()

    # Multiply the CPTs to get the joint distribution
    for vv in var_list_names:
        jd *= bn.cpt(vv)

    jd = jd.reorganize(var_list_names)  # Reorganize variables alphabetically
    jd_dict = {}

    # Loop over joint distribution to save (tuple, probability) in a dictionary
    for i in jd.loopIn():
        jd_dict[tuple(i.todict().values())] = jd.get(i)

    return jd_dict


# ____________________________________________________________________________________________________________________________________________________
def joint_dist_bn_dict_no_indicators(bn, var_indicator):
    """
    Compute the joint probability distribution excluding indicator variables.

    Parameters:
    - bn: The Bayesian network.
    - var_indicator: Dictionary of indicator variables.

    Returns:
    - A dictionary representing the joint probability distribution excluding indicators.
    """
    indicators_list = list(var_indicator.values())
    var_list_names = sorted(bn.names())

    for v in indicators_list:
        var_list_names.remove(v)

    jd = gum.Potential()
    
    for vv in var_list_names:
        jd *= bn.cpt(vv)

    jd = jd.reorganize(var_list_names)
    jd_dict = {}

    for i in jd.loopIn():
        jd_dict[tuple(i.todict().values())] = jd.get(i)

    return jd_dict


# ____________________________________________________________________________________________________________________________________________________
def compute_joint_probability_distribution_dict_prob(dataframe):
    """
    Compute the joint probability distribution from a DataFrame.

    Parameters:
    - dataframe: The DataFrame containing the data.

    Returns:
    - A dictionary representing the joint probability distribution.
    """
    ordered_columns = sorted(dataframe.columns)  # Order columns alphabetically
    dataframe = dataframe[ordered_columns]       # Select ordered columns
    total_samples = len(dataframe)

    # Group by all columns and count the occurrences
    counts = dataframe.groupby(list(dataframe.columns)).size().reset_index(name='count')

    # Calculate probabilities
    counts['probability'] = counts['count'] / total_samples

    # Convert to dictionary
    joint_probabilities = counts.set_index(list(dataframe.columns)).to_dict()['probability']

    return joint_probabilities


# ____________________________________________________________________________________________________________________________________________________
def study_pyAgrum_data_generation_quality(bns_list, file):
    """
    Generate and evaluate synthetic data quality using Wasserstein distance.

    Parameters:
    - bns_list: List of Bayesian networks or a single network.
    - file: Output file name to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if not isinstance(bns_list, list):
        bns_list = [bns_list]

    for bn in bns_list:
        bn_p_dict = joint_dist_bn_dict(bn)
        study_range = [10000, 1000000, 50000]
        res = []

        for i in range(*study_range):
            df = dg.generate_data(bn, i)
            df_p_dict = compute_joint_probability_distribution_dict_prob(df)
            kl_div = wasserstein_distance_(bn_p_dict, df_p_dict)
            res.append(kl_div)

        ax.plot(range(*study_range), res, label=f"Wasserstein dist ({os.path.basename(file)[:-4]}, Empirical dist(data))")

    ax.set_xlabel("Size of the database")
    ax.set_ylabel("KL Divergence")
    ax.set_title(f"Wasserstein distance ({os.path.basename(file)[:-4]}, Empirical dist(generated_data))")
    ax.legend()
    plt.savefig(file, format='png')
    plt.show()


# ____________________________________________________________________________________________________________________________________________________
def compute_rmse(complete, imputed, var_indicator):
    """
    Compute Root Mean Square Error (RMSE) for imputed values.

    Parameters:
    - complete: Complete dataset.
    - imputed: Imputed dataset.
    - var_indicator: Dictionary of variables with missingness indicators.

    Returns:
    - RMSE value.
    """
    key = next(iter(var_indicator))
    return np.sqrt(mean_squared_error(complete[key].to_numpy().astype(int), imputed[key].to_numpy().astype(int)))


# ____________________________________________________________________________________________________________________________________________________
def compute_macro_f1(complete, imputed, var_indicator):
    """
    Compute the macro F1-score for imputed values.

    Parameters:
    - complete: Complete dataset.
    - imputed: Imputed dataset.
    - var_indicator: Dictionary of variables with missingness indicators.

    Returns:
    - F1-score.
    """
    key = next(iter(var_indicator))
    return f1_score(complete[key].tolist(), imputed[key].tolist(), average='macro')


# ____________________________________________________________________________________________________________________________________________________
def kl_divergence(P, Q):
    """
    Compute the KL divergence between two probability distributions.

    Parameters:
    - P: Probability distribution 1.
    - Q: Probability distribution 2.

    Returns:
    - KL divergence value.
    """
    epsilon = 1e-20
    sum_s = 0

    for x, prob in P.items():
        if x in Q:
            sum_s += P[x] * np.log(P[x] / (Q[x] + epsilon))
        else:
            sum_s += P[x] * np.log(P[x] / epsilon)

    return sum_s


# ____________________________________________________________________________________________________________________________________________________
def euclidean_distance(dict1, dict2):
    """
    Compute the Euclidean distance between two dictionaries of probabilities.

    Parameters:
    - dict1: First probability dictionary.
    - dict2: Second probability dictionary.

    Returns:
    - Euclidean distance.
    """
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    sum_squared_diff = sum((dict1.get(key, 0) - dict2.get(key, 0)) ** 2 for key in all_keys)
    return math.sqrt(sum_squared_diff)


# ____________________________________________________________________________________________________________________________________________________
def number_different_tuples(dict1, dict2, db_size):
    """
    Compute the number of different tuples between two datasets.

    Parameters:
    - dict1: First dataset (as dictionary of counts).
    - dict2: Second dataset (as dictionary of counts).
    - db_size: The size of the database.

    Returns:
    - Number of different tuples.
    """
    all_keys = set(dict1.keys()).union(set(dict2.keys()))
    sum_diff = sum(abs(dict1.get(key, 0) - dict2.get(key, 0)) for key in all_keys)
    return db_size * sum_diff


# ____________________________________________________________________________________________________________________________________________________
def wasserstein_distance_(P, Q):
    """
    Compute the Wasserstein distance between two probability distributions.

    Parameters:
    - P: First probability distribution.
    - Q: Second probability distribution.

    Returns:
    - Wasserstein distance.
    """
    all_keys = set(P.keys()).union(set(Q.keys()))

    # Ensure both dictionaries have all keys
    for key in all_keys:
        P.setdefault(key, 0)
        Q.setdefault(key, 0)

    sorted_P = dict(sorted(P.items()))
    sorted_Q = dict(sorted(Q.items()))

    return wasserstein_distance(list(sorted_P.values()), list(sorted_Q.values()))
