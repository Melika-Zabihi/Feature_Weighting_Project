
## Description

This project is a classification task using K-Nearest Neighbors algorithm (KNN). The goal is to evaluate the performance of KNN algorithm with different feature weighting techniques. Four datasets have been used for this project: iris, digits, wine, and breast_cancer. Three feature weighting methods have been implemented for this project: random weighting, mutual information-based weighting, and label correlation-based weighting. 
   
## Datasets

Four datasets have been used for this project:

1. Iris: A dataset containing 150 samples of iris flowers with their sepal length, sepal width, petal length, and petal width (4 features).
2. Digits: A dataset containing 1797 images of handwritten digits (0-9) represented as 8x8 matrices (64 features).
3. Wine: A dataset containing 178 samples of wine with their chemical composition (13 features).
4. Breast Cancer: A dataset containing 569 samples of breast cancer with their 30 features.

## Feature Weighting Methods

Three feature weighting methods have been implemented for this project:

1. Random Weighting: Assigns random weights to each feature.
2. Mutual Information-Based Weighting: Calculates the mutual information between each feature and the target variable and assigns weights based on the mutual information scores.
3. Label Correlation-Based Weighting: Calculates the correlation between each feature and the target variable and assigns weights based on the correlation scores.
