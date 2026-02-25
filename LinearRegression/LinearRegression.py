import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_b = np.hstack([np.ones((n_samples, 1)), X])  # Add bias term
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # Normal equation
        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def train(self, X, y, learning_rate=0.01, n_iterations=1000):
        n_samples, n_features = X.shape
        self.coefficients = np.zeros(n_features)
        self.intercept = 0

        min_error = float('inf')
        for _ in range(n_iterations):
            y_pred = self.predict(X)
            cost = self.mean_squared_error(y, y_pred)
            if _ % 100 == 0:
                print(f"Iteration {_}: Cost {cost}")
            if cost < min_error:
                min_error = cost
            gradient = self.compute_gradient(y, y_pred, X)
            self.coefficients -= learning_rate * gradient[1:]
            self.intercept -= learning_rate * gradient[0]
        print(f"Final error: {min_error}")

    def predict(self, X):
        prediction = self.intercept + X @ self.coefficients
        return prediction
    
    def mean_squared_error(self, y_true, y_pred):
        n = len(y_true)
        mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
        return mse

    def compute_gradient(self, y_true, y_pred, X):
        n = len(y_true)
        m = len(self.coefficients)
        intercept = [sum((-2 * (y_true[i] - y_pred[i])) for i in range(n)) / n]
        gradient = np.array(intercept + [sum(-2 * X[i, j] * (y_true[i] - y_pred[i]) for i in range(n))/n for j in range(m)])
        return gradient