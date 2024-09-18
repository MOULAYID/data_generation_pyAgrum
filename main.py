import sys
import os
import pyAgrum as gum

# Add directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'quantitative_bayesian_networks', 'generators')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Imports from graphs_mcar
from graphs_mcar import (
    mcar_1,
    mcar_2,
    mcar_3
)

# Imports from graphs_mar
from graphs_mar import (
    mar_1,
    mar_2,
    mar_3,
    mar_4,
    mar_5,
    mar_6,
    mar_7,
    mar_8,
    mar_9,
    mar_10,
    mar_11
)

# Imports from graphs_mnar
from graphs_mnar import (
    recoverable_mnar_1,
    mnar_1,
    mnar_2,
    mnar_3,
    mnar_4,
    mnar_5,
    mnar_6,
    mnar_7,
    mnar_8,
    mnar_9,
    mnar_10,
    mnar_11,
    mnar_12,
    mnar_13,
    mnar_14,
    mnar_15,
    mnar_16
)


mnar_functions = [
    mnar_1,
    mnar_2,
    mnar_3,
    mnar_4,
    mnar_5,
    mnar_6,
    mnar_7,
    mnar_8,
    mnar_9,
    mnar_10,
    mnar_11,
    mnar_12,
    mnar_13,
    mnar_14,
    mnar_15,
    mnar_16
]

# List of functions to call for 'mar', with miss_rate as parameter
mar_functions = [
    mar_3,
    mar_4,
    mar_5,
    mar_6,
    mar_7,
    mar_8,
    mar_9,
    mar_10,
    mar_11
]

functions = mar_functions+mnar_functions
from src.data_generation import synthetic_data_experiment

def main():
    # Configuration Parameters
    host = "localhost"        # Database host address
    user = "postgres"         # Database username
    password = "postgres"     # Database password
    port = 5432               # Database port
    db_sizes_list = [30000,100000]           # The size of the dataset to be generated
    miss_rate_list = [0.1, 0.3]           # The rate at which values will be missing in the dataset
    storing_option = 0        # 0 = Save as CSV files, 1 = Save in Postgres, 2 = Save in both


    for miss_rate in miss_rate_list:
        # Bayesian_networks_list = [mcar_1(miss_rate), mar_1(miss_rate), recoverable_mnar_1(miss_rate)]  # mcar_1 is a model for generating missing values
        for Bayesian_newtwork in functions:
            bn, var_indicator, relation_name = Bayesian_newtwork(miss_rate)
            for db_size in db_sizes_list:
                synthetic_data_experiment(
                    host=host,
                    user=user,
                    password=password,
                    port=port,
                    database=relation_name,
                    bn=bn,
                    db_size=db_size,
                    var_indicator=var_indicator,
                    missing_rate=miss_rate,
                    storing_option=storing_option
                )

if __name__ == "__main__":
    main()
