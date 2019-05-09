from rnn_numpy import RNNNumpy
from datetime import datetime
import utils
import numpy as np
import os
import sys
root_path = os.path.dirname(os.path.realpath(__file__))


def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    losses = list()
    num_examples_seen = 0
    for epoch in range(nepoch):
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("{}: Loss after num_examples_seen={} epoch={}: {}".format(
                time, num_examples_seen, epoch, loss))
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate *= 0.5
                print("Setting learning rate to {}".format(learning_rate))
            sys.stdout.flush()

        for i in range(len(y_train)):
            model.numpy_sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1


if __name__ == '__main__':
    X_train = np.load(os.path.join(root_path, "data/X_train.npy"))
    y_train = np.load(os.path.join(root_path, "data/y_train.npy"))
    input = X_train[10]

    np.random.seed(0)
    vocabulary_size = 8000
    model = RNNNumpy(vocabulary_size)
    # o, s = model.forward(input)
    # print(s[0].shape)
    # print(o[0].T.shape)
    # print("Expected Loss for random predictions: {}".format(np.log(vocabulary_size)))
    # print("Actual Loss: {}".format(model.calculate_loss(X_train[:1000], y_train[:1000])))

    train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)