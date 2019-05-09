import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import datasets
import utils

# Generate a dataset and plot it
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

# # Train the logistic regression classifier
# clf = linear_model.LogisticRegressionCV()
# clf.fit(X, y)
# # Plot decision boundary
# utils.plot_decision_boundary(X, y, lambda x: clf.predict(x))

# Forward pass
def forward(model, X):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    a2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return W1, b1, W2, b2, a1, a2

# Loss
def calculate_loss(model, X):
    W1, b1, W2, b2, a1, a2 = forward(model, X)
    corect_logprobs = -np.log(a2[range(len(X)), y])
    data_loss = np.sum(corect_logprobs)
    # Add Regularization
    data_loss += reg_lambda / 2 * \
        (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1.0 / len(X) * data_loss

# backpropagate
def backpropagate(X, W2, a1, a2):
    delta3 = a2
    delta3[range(len(X)), y] -= 1
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)
    return dW1, db1, dW2, db2,

# Predict
def predict(model, X):
    _, _, _, _, _, a2 = forward(model, X)
    return np.argmax(a2, axis=1)

# Build Model
def build_model(X, reg_lambda, epsilon, nn_hdim, num_passess=20000, print_loss=False):
    # initialize the parameters to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # record the shared variables
    model = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    for i in range(num_passess):
        W1, b1, W2, b2, a1, a2 = forward(model, X)
        dW1, db1, dW2, db2 = backpropagate(X, W2, a1, a2)

        # regularize
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # update
        model['W1'] += -epsilon * dW1
        model['b1'] += -epsilon * db1
        model['W2'] += -epsilon * dW2
        model['b2'] += -epsilon * db2

        if print_loss and i % 1000 == 0:
            print("Loss after iteration {}: {}".format(
                i, calculate_loss(model, X)))
    return model


if __name__ == '__main__':
    nn_input_dim = 2
    nn_output_dim = 2

    epsilon = 0.01
    reg_lambda = 0.01
    nn_hdim = 3
    model = build_model(X, epsilon, reg_lambda, nn_hdim, print_loss=True)
    utils.plot_decision_boundary(X, y, lambda x: predict(model, x))
