import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
# gnb.configuration()
from pylab import *
import os
import pandas as pd
import numpy as np
import statistics
import csv 

# db import
from sqlalchemy import create_engine, types
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

import copy
import datetime
import time
import os
import cairosvg
import random
# ____________________________________________________________________________________________________________________________________________________

def connect_to_db(host, user, password, port, database):
    """
    Connects to a PostgreSQL server using the provided credentials.

    Parameters:
        host (str): The IP address or hostname of the PostgreSQL server.
        user (str): The username to connect to the PostgreSQL server.
        password (str): The password associated with the user to connect to the PostgreSQL server.
        port (int): The port number on which the PostgreSQL server listens for connections.
        database (str): The name of the database to connect to or create if it doesn't already exist.

    Returns:
        tuple: A tuple containing the connection object and the cursor object.
               The connection object can be used to execute SQL commands, and the cursor object
               is used to traverse the records from the result set.
    """
    # Connexion au serveur PostgreSQL
    conn = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        port=port)

    cursor = conn.cursor()
    conn.autocommit = True

    # Vérifier si la base de données existe déjà
    cursor.execute("SELECT datname FROM pg_catalog.pg_database WHERE datname = %s", (database,))
    if cursor.fetchone() == None:
        # Créer la base de données si elle n'existe pas
        cursor.execute("CREATE DATABASE "+database)
        print("Database created successfully........")
        cursor.close()
        conn.close()

    # Connexion à la base de données spécifiée
    print("connecting to database...........")
    conn = psycopg2.connect(
        database=database,
        host=host,
        user=user,
        password=password,
        port=port)

    cursor = conn.cursor()
    print("Connection successful!")
    return conn, cursor
# ____________________________________________________________________________________________________________________________________________________
def create_table_from_dataframe(df1, table_name, user, password, host, port, database, cur):
    """
    Creates a PostgreSQL table from a pandas DataFrame.

    Parameters:
        df1 (DataFrame): The pandas DataFrame containing the data to be inserted into the table.
        table_name (str): The name of the table to be created in the database.
        user (str): The username to connect to the PostgreSQL server.
        password (str): The password associated with the user to connect to the PostgreSQL server.
        host (str): The IP address or hostname of the PostgreSQL server.
        port (int): The port number on which the PostgreSQL server listens for connections.
        database (str): The name of the database where the table will be created.

    Returns:
        None
    """

    # Construct the database URI
    database_uri = 'postgresql://' + user + ':' + password + '@' + host + ':' + str(port) + '/' + database

    # Make a copy of the DataFrame and convert column names to lowercase
    df = df1.copy()
    df.columns = map(str.lower, df.columns)

    # Create a SQLAlchemy engine
    engine = create_engine(database_uri)

    # Get column names and types from the DataFrame
    columns = df.dtypes.reset_index()
    columns.columns = ['Column', 'Type']

    # Define mapping between pandas types and SQLAlchemy types
    sqlalchemy_types = {
        'int64': types.BigInteger,
        'int32': types.BigInteger,
        'int8': types.BigInteger,
        'int16': types.BigInteger,
        'integer': types.BigInteger,
        'float64': types.Float,
        'object': types.String,
        'datetime64[ns]': types.DateTime
        # Add more types as needed
    }

    # Map DataFrame types to SQLAlchemy types
    columns['Type'] = columns['Type'].map(str).map(sqlalchemy_types)

    # Create a dictionary for the columns and their data types
    dtype_dict = dict(zip(columns['Column'], columns['Type']))
    # print(dtype_dict)

    # Replace NaN values with None (SQL NULL)
    df = df.where(pd.notna(df), None)

    # Clean and format table name
    table_name = table_name.lower().replace(' ', '')


        ###############################to delete
    # cur.execute(f"DROP TABLE IF EXISTS {table_name};")
    cur.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table_name}');")
    table_exists = cur.fetchone()[0]

    if table_exists:
        # Delete all rows from the table
        cur.execute(f"DELETE FROM {table_name};")

    #     print(f"All rows from table '{table_name}' deleted successfully.")
    # else:
    #     print(f"Table '{table_name}' does not exist.")

        ###############################to delete

    # Create the table
    df.to_sql(table_name, engine, index=False, if_exists='append', dtype=dtype_dict)

    # Close the database connection
    # engine.dispose()
# ____________________________________________________________________________________________________________________________________________________

def show_graphical_info(bn):
    """
    This function displays the graphical information of a Bayesian Network (BN).

    Parameters:
    - bn: The Bayesian Network object to visualize.

    It shows the graphical representation of the BN,
    and displays the inferred marginal probabilities of all nodes in the network.
    """
    print("*" * 50)
    # Display the graphical representation of the BN
    gnb.showBN(bn, size='9')
    # Show the inferred marginal probabilities of all nodes in the network
    gnb.showInference(bn, size="10")
# ____________________________________________________________________________________________________________________________________________________
def generate_data(bn, db_size):
    """
    This function generates synthetic data from a Bayesian Network (BN).

    Parameters:
    - bn: The Bayesian Network object from which to generate data.
    - db_size: The size of the synthetic dataset to generate.

    Returns:
    - df: A Pandas DataFrame containing the generated data.
    """
    # Create a BNDatabaseGenerator object
    g = gum.BNDatabaseGenerator(bn)
    df = pd.DataFrame()
    # Generate synthetic data of the specified size
    g.drawSamples(db_size)
    # Convert the generated data to a Pandas DataFrame
    df = g.to_pandas()
    return df
# ____________________________________________________________________________________________________________________________________________________
def auto_convert_types(df):
    """
    Automatically converts columns in a DataFrame to appropriate data types.

    Parameters:
    - df (DataFrame): The DataFrame containing columns to be converted.

    Returns:
    - df (DataFrame): The DataFrame with converted data types.
    """
    for column in df.columns:
        try:
            # Attempt to convert to numeric, specifying downcast='integer'
            df[column] = pd.to_numeric(df[column], downcast='integer')
            # Check if the resulting dtype is 'int64'
            # if df[column].dtype == 'int64':
            #     # Mark the column as containing integers
            #     # print(f"Column '{column}' contains integers.")
            # else:
            #     # Mark the column as containing continuous values
            #     # print(f"Column '{column}' contains continuous values.")

        except ValueError:
            try:
                df[column] = pd.to_datetime(df[column])
                # Mark the column as containing datetime values
                # print(f"Column '{column}' contains datetime values.")
            except ValueError:
                # Handle other types as needed
                pass
    # print(">>>>>>>>>>>>>>>>>  ",df.dtypes)

    return df

# ____________________________________________________________________________________________________________________________________________________
def database_without_indicator(df, var_indicator):
    """
    Returns a DataFrame without columns indicated by the variable indicator.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - var_indicator (dict): A dictionary where keys are column names and values are indicators.
.
    Returns:
    - df_without_indicator (DataFrame): The DataFrame without the columns indicated by the variable indicator.
    """
    indicator_variables = list(var_indicator.values())
    return df.copy().drop(columns=indicator_variables, errors='ignore')
# ____________________________________________________________________________________________________________________________________________________

def introducing_NAs(df, var_indicator):
    """
    Introduces missing values (NaNs) into DataFrame based on a variable indicator.
    Rename the missing features by adding _star to each feature name

    Parameters:
    - df (DataFrame): The original DataFrame.
    - var_indicator (dict): A dictionary where keys are column names and values are indicators.

    Returns:
    - df1 (DataFrame): The DataFrame with introduced missing values.
    """
    df1 = df.copy()
    for var_key in var_indicator:
        var_star = var_key + '_star'
        df1.rename(columns={var_key: var_star}, inplace=True)
        df1.loc[df1[var_indicator[var_key]] == 1, var_star] = np.NAN

    return df1
# ____________________________________________________________________________________________________________________________________________________
def database_without_Vms_variables(df, var_indicator):
    """
    Returns a DataFrame without columns indicated by the variable indicator.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - var_indicator (dict): A dictionary where keys are column names and values are indicators.

    Returns:
    - df_without_Vms (DataFrame): The DataFrame without the partially observed variables (Vms).
    """
    part_obsrv_variables = list(var_indicator.keys())
    return df.copy().drop(columns=part_obsrv_variables, errors='ignore')
# ____________________________________________________________________________________________________________________________________________________
def save_output_files(title, tables_df, bn, size, timestamp_str, missing_rate, 
    var_indicator, storing_option,user,password,host,port,database):
  # results_list = all_answers_list.copy()

  current_dir = os.path.dirname(os.path.abspath(__file__))  
  project_root = os.path.abspath(os.path.join(current_dir, '..'))

  # the base directory
  base_directory = os.path.join(project_root, 'data', 'raw')
  print(title)
  # Create the base directory if it doesn't exist
  os.makedirs(base_directory, exist_ok=True)

  # Create subdirectories using os.path.join
  subdirectory = os.path.join(base_directory, title, str(size))

  # Create the subdirectories if they don't exist
  os.makedirs(subdirectory, exist_ok=True)

  # Example directory path
  directory_example = os.path.join(subdirectory, timestamp_str)
  os.makedirs(directory_example)

  svg_string = gnb.getBN(bn,size='24')


  # saving basic information about the incomplete db
  with open(directory_example+"/summary_incomplete_db.txt", 'w') as f:
    # Write the first few rows of the DataFrame
    f.write("Incomplete database summary:\n")

    # Write basic statistics
    f.write("Basic statistics:\n")
    stats_str = tables_df["incomplete_database"].describe(include='all').to_string()
    f.write(stats_str)
    f.write("\n\n")

    # Write missing values count
    f.write("Missing rate:\n")
    missing_values_str = (tables_df["incomplete_database"].isna().sum()/tables_df["incomplete_database"].shape[0]*100).to_string()
    f.write(missing_values_str)
    f.write("\n\n")

    # Write data types
    f.write("Data types:\n")
    dtypes_str = tables_df["incomplete_database"].dtypes.to_string()
    f.write(dtypes_str)
    f.write("\n")

  # storing the generated dbs 
  for table_name, table_df in tables_df.items():
    csv_file_path = f"{directory_example}/{table_name}.csv"

    # storing option 0: save as CSV files only
    if storing_option == 0:
        table_df.to_csv(csv_file_path, index=False)

    # storing option 1: save in the Postgres DB only
    elif storing_option == 1:
        # Connect to the database
        conn, cursor = connect_to_db(host, user, password, port, database)
        create_table_from_dataframe(table_df, table_name, user, password, host, port, database, cursor)
        print("closing connection ...................")
        # Close the database connection
        cursor.close()
        conn.close()

    # storing option 2: save in both CSV and Postgres DB
    elif storing_option == 2:
        table_df.to_csv(csv_file_path, index=False)
        # Connect to the database
        conn, cursor = connect_to_db(host, user, password, port, database)
        create_table_from_dataframe(table_df, table_name, user, password, host, port, database, cursor)
        print("closing connection ...................")
        # Close the database connection
        cursor.close()
        conn.close()


  # storing the graphical format of the Bayesian network
  output_png_path = directory_example+'/'+"bn.png"
  cairosvg.svg2png(bytestring=svg_string, write_to=output_png_path)

# error saving the bif file
  try:
    gum.saveBN(bn,directory_example+'/'+"bn.bif", True)
  except gum.FatalError as e:
    print(f"Error: {e}. Skipping the variable.")
    pass  # Skip the variable and continue with the next one
# ____________________________________________________________________________________________________________________________________________
def remove_star_suffix(df1):

  df = df1.copy()
  # identify those columns with _star
  star_columns = [col for col in df.columns if col.endswith('_star')]

  # Create a dictionary for renaming columns
  rename_dict = {col: col[:-5] for col in star_columns}

  # Rename columns
  df.rename(columns=rename_dict, inplace=True)

  return df
# ____________________________________________________________________________________________________________________________________________________
def keys_to_str(dictionary):
    if isinstance(dictionary, dict):
        return {tuple(map(str, k)): keys_to_str(v) for k, v in dictionary.items()}
    else:
        return dictionary
# ____________________________________________________________________________________________________________________________________________________
def conditional_probability(bn, full_obsrv_vars, proxy_vars, is_quantitative_mg, row):
    """
    Computes the conditional probability distribution based on a Bayesian Network (BN) for a given set of observed variables.

    Parameters:
    - bn: The Bayesian Network object used for inference.
    - full_obsrv_vars: A list of fully observed variables, which are used to compute the conditional probability.
    - proxy_vars: A list of proxy variables, which are used to condition the distribution.
    - is_quantitative_mg: A boolean indicating if the model is quantitative.
    - row: A dictionary where keys are variable names and values are their corresponding observations.

    Returns:
    - cond_dist: The conditional probability distribution given the observed variables.
    """
    
    # Initialize LazyPropagation for inference on the Bayesian Network
    ie = gum.LazyPropagation(bn)
    ie.makeInference()

    # Remove 'DuplicatesCount' from the set of fully observed variables
    full_obsrv_vars = set(full_obsrv_vars).difference(["DuplicatesCount"])

    # Create lists of full and partial observed variables without the '_star' suffix
    full_vars = [v.replace('_star', '') for v in full_obsrv_vars]
    partial_obsrv_vars = [v.replace("_star", "") for v in proxy_vars]

    # Create a list of indicators for partial observed variables (prefix with 'I')
    indicators_list = ['I' + v for v in partial_obsrv_vars]

    # Compute the conditional distribution based on whether the model is quantitative
    if is_quantitative_mg:
        cond_dist = ie.evidenceJointImpact(partial_obsrv_vars, full_vars + indicators_list)
        # Extract the distribution for cases where indicators are equal to 1
        for var in indicators_list:
            cond_dist = cond_dist.extract({var: 1})
    else:
        cond_dist = ie.evidenceJointImpact(partial_obsrv_vars, full_vars)

    # Extract the distribution for each fully observed variable based on its observed value
    for full_ob_var in full_obsrv_vars:
        var = full_ob_var.replace("_star", "")
        var_id = bn.variable(bn.idFromName(var))
        value_id = var_id.index(str(int(row[full_ob_var])))
        cond_dist = cond_dist.extract({var: value_id})

    return cond_dist
# ________________________________________________________________________________________________________________________
def compute_probabilistic_db(df, bn, var_indicator):
    """
    Compute a probabilistic database by defining super blocks based on tuples with missing values and their conditional probabilities.

    Parameters:
        df (DataFrame): The initial DataFrame containing tuples with potential missing values.
        bn (BayesianNetwork): A Bayesian Network model used for probability calculations.
        var_indicator (dict): A dictionary where keys are variable names and values indicate indicator variables.

    Returns:
        DataFrame: A DataFrame representing the probabilistic database with super blocks.
    """
    

    # Retrieve the first partially observed variable
    partial_obsrv_var = list(var_indicator.keys())[0]

    # Get the parent nodes of the partially observed variable
    parents_set = bn.parents(bn.idFromName(partial_obsrv_var))

    # Retrieve names of the parent nodes
    parents_names = [bn.variable(v).name() for v in parents_set]

    unique_df = df.drop_duplicates()
    # print("###########", unique_df)

    df_final = pd.DataFrame()

    # Iterate over rows with potentially missing values
    for index, row in unique_df.iterrows():
        print(f"defining the probability of tuples of pattern: {row}")
        prob_db = []
        # Identify which variables have missing values
        proxy_vars_with_missing_values = set(row.index[row.isnull()])
        # Identify which variables are fully observed
        fully_observed_vars = set(row.index[~row.isnull()])
        # List of variables with missing values
        proxy_vars = list(proxy_vars_with_missing_values)

        # Define the block and compute the probability if there are missing values
        if len(proxy_vars) > 0:
            # Calculate the conditional probability for the current tuple
            cp = conditional_probability(bn, fully_observed_vars, proxy_vars_with_missing_values,
                                          True, row)

            # Collect possible tuples
            possible_tuples = []
            for i in cp.loopIn():
                new_row = row.copy()
                for var in proxy_vars:
                    value = bn.variable(bn.idFromName(var.replace("_star", ""))).label(i[var.replace("_star", "")])
                    # Update row with values for proxy variables
                    if pd.api.types.is_float_dtype(df.dtypes[var]):
                        new_row[var] = float(value)
                    else:
                        new_row[var] = value
                new_row["probability"] = cp.get(i)
                # new_row["block_id"] = index

                possible_tuples.append(new_row)
            # Append all possible tuples to the prob_db list
            prob_db.extend(possible_tuples)

        else:
            # If no missing values, assign a probability of 1
            row["probability"] = 1
            # row["block_id"] = index
            prob_db.append(row)

        prob_db_df = pd.DataFrame(prob_db)

        # Define the columns to ignore in the row
        columns_to_ignore = ['probability']

        # Subset the row by dropping the columns that are not in the DataFrame
        filtered_row = row.drop(labels=columns_to_ignore, errors='ignore')

        # Compare the DataFrame rows with the filtered row
        identical_rows = df[df.apply(lambda x: x.equals(filtered_row), axis=1)]

        # Select rows to modify
        rows_to_modify = identical_rows.index

        for row_id in rows_to_modify:
            # Create a copy of prob_df
            prob_df_copy = prob_db_df.copy()

            # Add or modify the 'block_id' column with the current row_id
            prob_df_copy['block_id'] = row_id

            # Append the modified prob_df_copy to df_final
            df_final = pd.concat([df_final, prob_df_copy], ignore_index=True)

    # Rearrange the 'block_id' column to be the first column
    block_id_col = df_final.pop("block_id")
    df_final.insert(0, 'block_id', block_id_col)

    return remove_star_suffix(auto_convert_types(df_final.sort_values(by='block_id', ignore_index=True)))
# ________________________________________________________________________________
def convert_row_types(row):
    converted_row = row.copy()  # Create a copy of the original row
    for idx, value in row.iteritems():
        try:
            converted_value = float(value)  # Convert to float
            if not np.isnan(converted_value):  # Check if the value is not NaN
                converted_value = int(converted_value)  # Convert to integer
            converted_row[idx] = converted_value
        except ValueError:
            pass  # If unable to convert, keep it as it is
    return converted_row

# __________________________________________________________________________________________________________________________________________________
def databases_generator(bn, var_indicator, db_size):
    """
    Generate and process different versions of a database using a Bayesian Network model.

    Parameters:
        bn (BayesianNetwork): The Bayesian Network model used for data generation and probability computations.
        var_indicator (dict): Dictionary indicating indicator variables for handling missing values.
        db_size (int): The desired size of the generated database.

    Returns:
        dict: A dictionary containing the complete database, incomplete database, expanded format, and probabilistic database.
    """

    print("Start generating the initial data by pyAgrum")
    # Generate an initial dataset using the Bayesian Network model
    df = generate_data(bn, db_size)
    print("pyAgrum has finished generating the initial data")
    
    # Convert data types in the DataFrame to appropriate types
    df = auto_convert_types(df)
    print(df)

    # Create a complete database by removing indicator variables
    complete_database = database_without_indicator(df.copy(), var_indicator)
    print("The complete database\n", complete_database)

    print("Generating the incomplete database D*")

    # Introduce missing values into the DataFrame according to the indicator variables
    df_all_database_information = introducing_NAs(df.copy(), var_indicator)
    print("The expanded format of the database\n", df_all_database_information)

    # Remove variables related to VMs and create an incomplete database
    incomplete_database = database_without_indicator(
        database_without_Vms_variables(df_all_database_information, var_indicator), 
        var_indicator
    )
    print("The incomplete database\n", incomplete_database)

    print("computing the Probabilistic database")
    # Compute the probabilistic database from the incomplete database
    pdb = compute_probabilistic_db(incomplete_database.copy(), bn, var_indicator)
    print("Probabilistic database\n", pdb)

    # Return a dictionary containing all versions of the database
    return {
        "complete_database": complete_database,
        "incomplete_database": incomplete_database,
        "expanded_format": df,
        "prob_db": pdb
    }

# ____________________________________________________________________________________________________________________________________________________

def load_BN_from_bif(bn_path):
    return gum.loadBN(bn_path)
# ____________________________________________________________________________________________________________________________________________________

def synthetic_data_experiment(host, user, password, port, database, bn, db_size, var_indicator, missing_rate, storing_option):
    """
    Conduct an experiment by generating synthetic data, adding missing values, and storing the results in a database.

    Parameters:
        host (str): The hostname of the database server.
        user (str): The username for database authentication.
        password (str): The password for database authentication.
        port (int): The port number on which the database server is listening.
        database (str): The name of the database where the data will be stored.
        bn (BayesianNetwork): The Bayesian Network model used to generate the synthetic data.
        db_size (int): The size of the synthetic dataset to be generated.
        var_indicator (dict): Dictionary indicating indicator variables for handling missing values.
        missing_rate (float): The rate of missingness to be applied to the data.
        storing_option (str): Option specifying where to store the generated data.

    Returns:
        None
    """

    # Print message indicating the start of the experiment with details
    print(f"=============>>> running example {database} with missingness rate = {missing_rate}")

    # Prepare the database name with a timestamp for uniqueness
    relation_name = database
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    database = f"{relation_name}_{timestamp_str}".lower().replace('-', '').replace(' ', '')



    # Display graphical information of the Bayesian Network
    show_graphical_info(bn)

    # Generate synthetic data using the Bayesian Network model and specified parameters
    dbs = databases_generator(bn, var_indicator, db_size)

    # Save the generated data and other relevant information to the database
    save_output_files(
        relation_name, dbs, bn, db_size, timestamp_str, missing_rate, 
        var_indicator, storing_option, user, password, host, port, database)

    # Print message indicating the completion of the process
    print("done!")
    
