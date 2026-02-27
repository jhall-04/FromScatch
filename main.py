from LinearRegression.LinearRegression import LinearRegression
from LogisticRegression.LogisticRegression import LogisticRegression
import pandas as pd
import numpy as np

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
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def mlp_example():
    pass
def main():
    # linear_regression_example()
    # logistic_regression_example()
    mlp_example()

if __name__ == "__main__":
    main()
