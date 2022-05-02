import torch
import numpy as np

# get the ith fold data
def get_k_fold_data(k, i, X, Y):
    # n: numbers of samples of one fold
    n = X.shape[0] // k
    X_train, Y_train = None, None
    X_test, Y_test = None, None
    first = 1

    for j in range(k):
        # slice not include right endpoint
        index = slice(n*j, n*(j+1))

        X_part, Y_part = X[index, :], Y[index]

        if j == i:
            X_test, Y_test = X_part, Y_part
        elif first:
            first = 0
            X_train, Y_train = X_part, Y_part
        else:
            X_train = np.concatenate([X_train, X_part], axis=0)
            Y_train = np.concatenate([Y_train, Y_part], axis=0)

    return X_train, Y_train, X_test, Y_test





