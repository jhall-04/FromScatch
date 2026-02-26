import numpy as np

class LogisticRegression:
    def __init__(self):
        self.coefficients = None
        self.bias = None
        self.z = None

    def sigmoid(self):
        return 1 / (1 + np.exp(-self.z))
    
    def predict(self, X):
        self.z = X @ self.coefficients + self.bias
        y_pred = self.sigmoid()
        return np.where(y_pred >= 0.5, 1, 0)
    
    def train(self, X, y, learning_rate=0.01, iterations=1000):
        num_samples, num_features = X.shape
        self.coefficients = np.zeros(num_features)
        self.bias = 0
        self.z = np.zeros(num_samples)

        min_cost = float('inf')
        min_coefficients = None
        min_bias = None
        for i in range(iterations):
            y_pred = self.predict(X)
            cost = self.log_likelihood(y, y_pred)
            if i % 100 == 0:
                print(f"Iteration {i}: Cost {cost}")
            if cost < min_cost:
                min_cost = cost
                min_coefficients = self.coefficients.copy()
                min_bias = self.bias
            else:
                pass

    def log_likelihood(self, y, y_pred):
        n = len(y)
        likelihood = -np.sum(y * np.log(self.sigmoid()) + (1 - y) * np.log(1 - self.sigmoid()) for i in range(n)) / n
        return likelihood
    
    def compute_gradient(self, X, y, y_pred):
        pass