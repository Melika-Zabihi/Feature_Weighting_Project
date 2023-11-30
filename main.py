import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN
cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

def main():
    dataset = datasets.load_iris()
    # dataset = datasets.load_digits()
    # dataset = datasets.load_wine()
    # dataset = datasets.load_breast_cancer()

    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    knn = KNN(k=5)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    acc = calculate_accuracy(predictions, y_test)
    print(acc)


def calculate_accuracy(predictions, y_test):
    return np.sum(predictions == y_test) / len(y_test)


if __name__ == "__main__":
    main()