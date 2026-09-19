import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, X, y):
        self.X = X          # design matrix
        self.y = y          # target label
        self.n = y.shape[0] # number of samples
        self.d = X.shape[1] # the length of the weight

    #Calculate MSE loss
    def compute_loss(self, w, b): 
        y_pred = self.X @ w + b                   # y_pred->predicted values,w->weight,b->bias
        mse = np.mean(np.square(y_pred - self.y))
        return mse
    
    #Calculate gradient of w and b
    def compute_gradient(self, w, b):
        y_pred = self.X @ w + b 
        d_w = (2.0 / self.n) * self.X.T @ (y_pred - self.y) # derivative of w
        d_b = (2.0 / self.n) * np.sum(y_pred - self.y)     # derivative of b
        return d_w, d_b


class GradientDescent:
    def __init__(self, lr=0.005, epochs=80000): #lr->learning rate
        self.lr = lr 
        self.epochs = epochs

    def optimize(self, loss): 
        w = np.zeros(loss.d)  # Initialize weights as zero vector 
        b = 0.0

        for epoch in range(self.epochs):
            d_w, d_b = loss.compute_gradient(w, b)  
            # Update parameters by gradient descent rule
            w = w - self.lr * d_w
            b = b - self.lr * d_b

            if epoch % 200 == 0:
                current_loss = loss.compute_loss(w, b)
                print(f"Epoch {epoch:5d} | Loss = {current_loss:.4f}")
        return w, b


if __name__ == "__main__":

    concrete = fetch_ucirepo(id=165)   # Load concrete compressive strength dataset (UCI id=165)
    X_raw = concrete.data.features.values
    y_raw = concrete.data.targets.values.ravel()

    print("Dataset name:", concrete.metadata.name)
    print("Number of samples:", X_raw.shape[0])
    print("Number of features:", X_raw.shape[1])

    X = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0)  # Feature standardization

    # Split into training set and test set，test size 20%
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    y_train, y_test = train_test_split(y_raw, test_size=0.2, random_state=42)

    loss = LinearRegression(X_train, y_train)
    optimizer = GradientDescent(lr=0.01, epochs=15000)  # Create gradient descent optimizer
    w_opt, b_opt = optimizer.optimize(loss)  

    print("=== Training Finished ===")
    print("Optimal weights w:", w_opt)
    print("Optimal bias b:", b_opt)

    train_loss = loss.compute_loss(w_opt, b_opt)
    print("Training set final loss:", train_loss)

    test_model = LinearRegression(X_test, y_test)
    test_loss = test_model.compute_loss(w_opt, b_opt)
    print("Test set final loss:", test_loss)
