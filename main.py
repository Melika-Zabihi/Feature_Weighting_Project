import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN
from sklearn.feature_selection import mutual_info_classif
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
import enum
import pandas as pd

def main():

    # dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()

    X, y = dataset.data, dataset.target
    X = apply_weights(X, y, Weight.ff_correlation)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    knn = KNN(k=7)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc = calculate_accuracy(predictions, y_test)
    print(acc)


def calculate_accuracy(predictions, y_test):
    return (np.sum(predictions == y_test) / len(y_test))*100


def apply_weights(X, y, weight_type):

    if weight_type == Weight.normal:
        X = normalize(X)
        return X

    if weight_type == Weight.random:
        random_matrix = np.random.rand(*X.shape)
        weighted_x = np.round(X * random_matrix, 2)
        normalized_x = normalize(weighted_x)
        return normalized_x

    if weight_type == Weight.fl_correlation:
        mi = mutual_info_classif(X, y)
        weighted_x = X * mi
        normalized_x = normalize(weighted_x)
        return normalized_x

    if weight_type == Weight.ff_correlation:
        df = pd.DataFrame(data=X)
        correlation_matrix = df.corr().abs()
        avg_corr = (correlation_matrix.sum() - 1) / (len(correlation_matrix.columns) - 1)
        reciprocal_avg_corr = 1 / avg_corr
        result = df * reciprocal_avg_corr
        final_result = result.to_numpy()
        normalized_result = normalize(final_result)
        return normalized_result


def plot_data(X, data):
    df = pd.DataFrame(np.c_[X, data.target],
                      columns=np.append(data['feature_names'], ['target']))

    feature1 = df.columns[0]
    feature2 = df.columns[1]

    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature1], df[feature2], c=df['target'], cmap=cmap)
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()


def normalize(series):
   return (series - series.min()) / (series.max() - series.min())


class Weight(enum.Enum):
    normal = 1
    random = 2
    fl_correlation = 3
    ff_correlation = 4


if __name__ == "__main__":
    main()