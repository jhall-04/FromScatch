from LinearRegression.LinearRegression import LinearRegression
import pandas as pd
import numpy as np

def main():
    lr_model = LinearRegression()
    data = pd.read_csv("LinearRegression/hou_all.csv")
    data = data.drop(columns=["BIAS_COL"])
    data = data.to_numpy()
    X = data[:, :-1]  # Features
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    print(f"X: {X[0]}")
    y = data[:, -1]   # Target variable
    lr_model.train(X, y)
    print("Coefficients:", lr_model.coefficients)
    print("Intercept:", lr_model.intercept)

if __name__ == "__main__":
    main()
