import numpy as np

class LogisticRegression:
    def __init__(self):
        self.coefficients = None
        self.bias = None
        self.z = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        z = X @ self.coefficients + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        y_pred = self.predict_proba(X)
        return np.where(y_pred >= 0.5, 1, 0)
    
    def train(self, X, y, learning_rate=0.01, iterations=1000, early_stopping=5):
        num_samples, num_features = X.shape
        self.coefficients = np.zeros(num_features)
        self.bias = 0.0
        stop = 0

        max_cost = float('-inf')
        best_coefficients = None
        best_bias = None
        for i in range(iterations):
            y_pred = self.predict_proba(X)
            cost = self.log_likelihood(y, y_pred)
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")
            if cost > max_cost:
                max_cost = cost
                best_coefficients = self.coefficients.copy()
                best_bias = self.bias
                stop = 0
            else:
                stop += 1
                if stop > early_stopping:
                    print(f"Early stopping at iteration {i} with cost {cost}")
                    break
            gradient = self.compute_gradient(X, y, y_pred)
            self.coefficients -= learning_rate * gradient[1:]
            self.bias -= learning_rate * gradient[0]
        self.coefficients = best_coefficients
        self.bias = best_bias

    def log_likelihood(self, y, y_pred):
        n = len(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        likelihood = sum(y[i] * np.log(y_pred[i]) + (1 - y[i]) * np.log(1 - y_pred[i]) for i in range(n)) / n
        return likelihood
    
    def compute_gradient(self, X, y, y_pred):
        bias_gradient = np.sum(y_pred - y) / len(y)
        coefficients_gradient = np.array([sum(X[i][j] * (y_pred[i] - y[i]) for i in range(len(y))) for j in range(int(X.shape[1]))]) / len(y)
        return np.hstack([bias_gradient, coefficients_gradient]) 