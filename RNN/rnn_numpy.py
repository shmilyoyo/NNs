import numpy as np
import utils

class RNNNumpy(object):
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Initialize parameters
        self.U = np.random.uniform(-np.sqrt(1.0 / word_dim), np.sqrt(1.0 / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1.0 / hidden_dim), np.sqrt(1.0 / hidden_dim), (hidden_dim, hidden_dim))
    
    def forward(self, x):
        T = len(x)
        s = np.zeros((T + 1, self.hidden_dim))
        s[-1] = np.zeros(self.hidden_dim)
        o = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            s[t] = np.tanh(self.U[:, x[t]] + self.W.dot(s[t - 1]))
            o[t] = utils.softmax(self.V.dot(s[t]))
        return [o, s]

    def predict(self, x):
        o, s = self.forward(x)
        return np.argmax(o, axis=1)

    def calculate_total_loss(self, x, y):
        L = 0
        # For each sentence
        for i in np.arange(len(y)):
            o, s = self.forward(x[i])
            # Only care about our prediction of the correct words
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1 * np.sum(np.log(correct_word_predictions))
        return L
    
    def calculate_loss(self, x, y):
        N = np.sum(len(y_i) for y_i in y)
        return self.calculate_total_loss(x, y) / N
    
    def bptt(self, x, y):
        T = len(y)
        # forward
        o, s = self.forward(x)
        # calculate gradients
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        # each output backwards
        for t in np.arange(T)[::-1]:
            dV += np.outer(delta_o[t], s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            for bptt_step in np.arange(max(0, t - self.bptt_truncate), t + 1)[::-1]:
                dW += np.outer(delta_t, s[bptt_step - 1])
                dU[:, x[bptt_step]] += delta_t
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step - 1] ** 2)
        return [dU, dV, dW]
    
    def numpy_sgd_step(self, x, y, learning_rate):
        dU, dV, dW = self.bptt(x, y)
        self.U -= learning_rate * dU
        self.V -= learning_rate * dV
        self.W -= learning_rate * dW

            
