import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN
from sklearn.feature_selection import mutual_info_classif
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
import enum
import pandas as pd

def main():

    # dataset = datasets.load_iris()
    dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()
    # dataset = datasets.load_diabetes()

    X, y = dataset.data, dataset.target
    X = apply_weights(X, y, Weight.fl_correlation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    acc = calculate_accuracy(predictions, y_test)
    print(acc)


def calculate_accuracy(predictions, y_test):
    return np.sum(predictions == y_test) / len(y_test)


def apply_weights(X, y, weight_type):

    if weight_type == Weight.normal:
        return X * 1

    if weight_type == Weight.random:
        random_matrix = np.random.rand(*X.shape)
        weigthed_X = np.round(X * random_matrix, 2)
        return weigthed_X

    if weight_type == Weight.fl_correlation:
        mi = mutual_info_classif(X, y)
        weigthed_X = X * mi
        return weigthed_X

    if weight_type == Weight.ff_correlation:
        df = pd.DataFrame(data=X)
        correlation_matrix = df.corr()
        avg_corr = (correlation_matrix.sum() - 1) / (len(correlation_matrix.columns) - 1)
        reciprocal_avg_corr = 1 / avg_corr
        result = df * reciprocal_avg_corr
        final_result = result.to_numpy()
        return final_result


class Weight(enum.Enum):
    normal = 1
    random = 2
    fl_correlation = 3
    ff_correlation = 4

if __name__ == "__main__":
    main()