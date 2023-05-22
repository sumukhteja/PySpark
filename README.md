# PySpark Machine Learning Examples

This repository contains PySpark code examples for machine learning tasks. The examples demonstrate how to use PySpark to train a logistic regression model and make predictions on different datasets.

## Code Example 1: Wine Classification

The first code example demonstrates how to perform wine classification using PySpark. The dataset used in this example is the Wine Recognition Dataset from the UCI Machine Learning Repository.

The code performs the following steps:
1. Creates a SparkSession.
2. Defines the schema for the wine dataset.
3. Reads the dataset from a remote URL.
4. Prepares the data by assembling the feature columns.
5. Splits the data into training and test sets.
6. Trains a logistic regression model on the training set.
7. Makes predictions on the test set.
8. Evaluates the model's accuracy.

## Code Example 2: Iris Dataset Classification

The second code example demonstrates how to perform classification on the Iris dataset using PySpark.

The code performs the following steps:
1. Creates a SparkSession.
2. Downloads the Iris dataset from a remote URL.
3. Loads the dataset into a DataFrame.
4. Converts the "species" column from string to numeric.
5. Prepares the data by assembling the feature columns.
6. Splits the data into training and test sets.
7. Trains a logistic regression model on the training set.
8. Makes predictions on the test set.
9. Evaluates the model's accuracy.
