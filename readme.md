# Ensure that PostgreSQL is installed on your machine.
# Install the required Python packages.
```bash
pip install -r requirements.txt 
```
#### If you encounter connection issues to PostgreSQL, ensure to update the parameters in the main such as user and password according to your PostgreSQL configuration.



# **Data Generation with pyAgrum: Simulating Incomplete Datasets**

## **Project Overview**

This project leverages **pyAgrum**, a Python library for Bayesian networks, to generate synthetic data with controlled missingness mechanisms. The input to the project is a **Bayesian network (BN)** that represents:
1. The causal structure of the attributes in a dataset.
2. The mechanism of missingness, encoding the probability of missing values.

By combining causal relationships between variables and the missingness mechanism, this project creates both **complete datasets** and **incomplete datasets** by introducing missing values. Then compute a block independent probabilistic database from the incomplete database by leveraging the bayesian network.

## **Inputs**

1. **Missingness Graph (Bayesian Network)**:
   - A **Bayesian network (BN)** is provided as input, which defines the dependencies between variables and the missingness mechanism using *indicator variables*.
   - Each indicator variable determines whether an attribute will have a missing value (`NA`) in the final dataset.

2. **Database Size**:
   - User specifies the number of tuples in the generated synthetic dataset.

3. **Missingness Rate**:
   - The probability of missingness is determined by the indicator variables in the missingness graph.

## **Data Generation Process**

### 1. **Complete Data Generation**
   - Using pyAgrum, a **complete synthetic dataset** is generated based on the input BN, ensuring dependencies between attributes are respected according to the conditional/absolute probability distributions.

### 2. **Handling Missing Data Using Indicators**
   - Each partially observed attribute has a corresponding **indicator variable** that determines whether a missing value (`NA`) should be introduced in the dataset.
   - If the indicator variable equals 1, the target attribute is set to `NA`.

### 3. **Probabilistic Database Generation**
   - Using the **incomplete database**, and the conditional/absolute probabilities defined in the BN, we build the probabilistic database.

## **Outputs**

1. **Complete Dataset**: A synthetic dataset without missing values.
2. **Incomplete Dataset**: The same dataset with missing values (`NA`) introduced according to the missingness graph.
3. **Probabilistic Dataset**: .

## **How to Use**

