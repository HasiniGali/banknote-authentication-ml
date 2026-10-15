# Banknote Authentication Using Machine Learning

## Project Overview

This project uses machine learning to classify banknotes into two
classes using numerical features extracted from banknote images.

Two machine learning algorithms are implemented:

- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

The models are trained and evaluated on the UCI Banknote Authentication
dataset.

## Problem Statement

Counterfeit banknotes can cause financial losses. Manually identifying
fake notes can be difficult, so machine learning can help classify
banknotes based on patterns in their extracted features.

This project demonstrates a basic machine learning approach to
banknote classification.

## Dataset

- Name: UCI Banknote Authentication
- Source: UCI Machine Learning Repository
- Records: 1,372
- Input features: 4
- Target: Class label

Dataset link:
https://archive.ics.uci.edu/dataset/267/banknote+authentication

Features:

1. Variance
2. Skewness
3. Curtosis
4. Entropy

Note: This dataset is not specifically an Indian-currency dataset.
The project demonstrates the classification approach using the
available public dataset.

## Machine Learning Models

### 1. Decision Tree

A Decision Tree learns rules from the training data to classify
banknotes into different classes.

### 2. K-Nearest Neighbors

KNN classifies a banknote by comparing its features with similar
examples in the training dataset.

Feature scaling is applied before KNN training.

## Methodology

1. Load the dataset.
2. Inspect and prepare the data.
3. Split the data into training and testing sets.
4. Train the Decision Tree model.
5. Train the KNN model.
6. Evaluate both models.
7. Generate confusion matrices.
8. Compare model accuracy.
9. Save the trained models.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- UCI Machine Learning Repository

## Project Structure

```text
banknote-authentication-ml/
├── train_model.py
├── requirements.txt
├── README.md
├── models/
└── results/