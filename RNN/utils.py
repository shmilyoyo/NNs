import numpy as np

def softmax(X):
    Xt = np.exp(X - np.max(X))
    return Xt / np.sum(Xt)