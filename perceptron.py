import numpy as np
from tqdm import tqdm


def unit_step_func(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate: float = 0.1, n_iters: int =1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # set y values either 0 or 1
        y_ = np.array([1 if i > 0 else 0 for i in y])

        for _ in tqdm(range(self.n_iters), 
                    desc="Fitting the perceptron...", 
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = unit_step_func(linear_output)


                # Perceptron update rule:
                # -1 (y_pred is too high) -> decrease weights
                #  1 (y_pred is too low)  -> increase weights
                #  0 (correct)            -> no change
                update = y_[idx] - y_predicted

                self.weights += self.lr * update * x_i
                self.bias += self.lr * update


    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = unit_step_func(linear_output)
        return y_predicted 


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    X, y = datasets.make_blobs(
        n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123
    )

    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions = p.predict(X_test)

    print("Perceptron classification accuracy", accuracy(y_test, predictions))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

    x0_1 = np.amin(X_train[:, 0])
    x0_2 = np.amax(X_train[:, 0])

    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

    ax.plot([x0_1, x0_2], [x1_1, x1_2], "k")

    ymin = np.amin(X_train[:, 1])
    ymax = np.amax(X_train[:, 1])
    ax.set_ylim([ymin - 3, ymax + 3])

    plt.show()