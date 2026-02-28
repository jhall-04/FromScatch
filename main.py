from LinearRegression.LinearRegression import LinearRegression
from LogisticRegression.LogisticRegression import LogisticRegression
from MLP.MLP import MLP

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

def linear_regression_example():
    lr_model = LinearRegression()
    data = pd.read_csv("LinearRegression/hou_all.csv")
    data = data.drop(columns=["BIAS_COL"])
    data = data.to_numpy()
    X = data[:, :-1]  # Features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    print(f"X: {X[0]}")
    y = data[:, -1]   # Target variable
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
    lr_model.train(X_train, y_train)
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)
    y_pred = lr_model.predict(X_test)
    print("R^2 Score:", lr_model.score(y_test, y_pred))

def logistic_regression_example():
    lr_model = LogisticRegression()
    data = pd.read_csv("LogisticRegression/wdbc.data", header=None)
    data = data.drop(columns=[0])  # Drop ID column
    y = data[1].apply(lambda x: 1 if x == "M" else 0).to_numpy()  # Convert to binary labels
    X = data.drop(columns=[1]).to_numpy()  # Features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardize features
    X_train, X_test = X[:int(0.8 * len(X))], X[int(0.8 * len(X)):]
    y_train, y_test = y[:int(0.8 * len(y))], y[int(0.8 * len(y)):]
    lr_model.train(X_train, y_train)
    print("Coefficients:", lr_model.coefficients)
    print("Bias:", lr_model.bias)
    y_pred = lr_model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("Accuracy:", accuracy)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def convert_to_greyscale(X):
    n = X.shape[0]
    X_grey = np.zeros((n, 32*32))
    for i in range(n):
        X_grey[i] = np.dot(X[i].reshape(3, 32, 32).transpose(1, 2, 0)[...,:3], [0.213, 0.715, 0.072]).flatten()
    return X_grey / 255.0

def mlp_example():
    train = unpickle("MLP/data_batch_1")
    print(train.keys())
    test = unpickle("MLP/test_batch")
    x_train = train[b'data']
    y_train = train[b'labels']
    x_test = test[b'data']
    y_test = test[b'labels']
    print(f"x_train shape: {x_train.shape}, y_train shape: {len(y_train)}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {len(y_test)}")
    x_train = convert_to_greyscale(x_train)
    x_test = convert_to_greyscale(x_test)
    fig, axs = plt.subplots(2, 4, figsize=(15, 10))
    for i in range(8):
        img = x_train[i].reshape(32, 32)
        axs[i // 4, i % 4].imshow(img, cmap='gray')
        axs[i // 4, i % 4].set_title(f"Label: {y_train[i]}")
        axs[i // 4, i % 4].axis('off')
    plt.savefig("mlp_example.png")
    

def main():
    # linear_regression_example()
    # logistic_regression_example()
    mlp_example()

if __name__ == "__main__":
    main()
