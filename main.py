from LinearRegression.LinearRegression import LinearRegression
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

def main():
    linear_regression_example()

if __name__ == "__main__":
    main()
