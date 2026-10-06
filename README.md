# Iris Flower Classification

## Overview

This project uses a Decision Tree machine learning model to classify Iris flowers into three species:

- Setosa
- Versicolor
- Virginica

The model uses four measurements:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Dataset

The project uses the built-in Iris dataset from scikit-learn.

The dataset contains 150 Iris flower samples across three species.

## Machine Learning Process

The project follows these steps:

1. Load the Iris dataset
2. Separate the features and target
3. Split the data into training and testing sets
4. Train a Decision Tree classifier
5. Make predictions on the test data
6. Evaluate the model using accuracy, confusion matrix, precision, recall and F1-score

The data was split using 80% for training and 20% for testing.

## Results

The Decision Tree achieved 100% accuracy on the test dataset.

The confusion matrix showed that all 30 test samples were classified correctly.

## Technologies

- Python
- pandas
- scikit-learn
- matplotlib
- Jupyter Notebook

## Project Files

- `iris_model.ipynb` — Jupyter Notebook containing the analysis and model
- `requirements.txt` — Python dependencies
- `README.md` — Project documentation

## How to Run

Install the required packages using:

`pip install -r requirements.txt`

Then open `iris_model.ipynb` in Jupyter Notebook and run the cells.
