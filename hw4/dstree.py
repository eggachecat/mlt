import numpy as np
import time


class BranchNode:
    def __init__(self, ts, fi):
        self.left = None
        self.right = None
        self.ts = ts
        self.fi = fi

    def predict(self, x):

        if x[self.fi] < self.ts:
            if isinstance(self.left, BranchNode):
                return self.left.predict(x)
            else:
                return self.left
        else:
            if isinstance(self.right, BranchNode):
                return self.right.predict(x)
            else:
                return self.right


def fit(train_x, train_y, free_depth):
    n_data, n_dim = train_x.shape

    if free_depth == 0 or np.sum(np.asmatrix(train_y) == train_y[0]) == len(train_y):
        unique, counts = np.unique(train_y, return_counts=True)
        return unique[np.argsort(counts)[-1]]

    best_ts = None
    best_impurity = float("inf")
    best_index = -1
    best_l_is = []
    best_r_is = []

    for i in range(n_dim):

        feature_vector = train_x[:, i]

        sorted_feature_vector = np.sort(feature_vector, 0)

        theta_arr = (np.r_[sorted_feature_vector[0:1, :] - 0.1, sorted_feature_vector] + np.r_[
            sorted_feature_vector, sorted_feature_vector[-1:, :] + 0.1]) / 2

        for theta in theta_arr:

            states = feature_vector < theta
            l_is, r_is = np.where(states)[0], np.where(states == False)[0]
            l_y = train_y[l_is]
            r_y = train_y[r_is]

            if len(l_y) == 0:
                l_gini = 1
            else:
                l_gini = len(l_y) - (np.sum(l_y == -1) ** 2 + np.sum(l_y == 1) ** 2) / len(l_y)

            if len(r_y) == 0:
                r_gini = 1
            else:
                r_gini = len(r_y) - (np.sum(r_y == -1) ** 2 + np.sum(r_y == 1) ** 2) / len(r_y)

            impurity = l_gini + r_gini

            if impurity < best_impurity:
                best_impurity = impurity
                best_ts = theta
                best_index = i
                best_l_is = l_is
                best_r_is = r_is
    #
    # print("best_theta is", best_ts)
    # print("best_gini is", best_impurity)
    # print("best_index is", best_index)

    bn = BranchNode(best_ts, best_index)

    bn.left = fit(train_x[best_l_is], train_y[best_l_is], free_depth - 1)
    bn.right = fit(train_x[best_r_is], train_y[best_r_is], free_depth - 1)

    return bn



def read_input(file_path, segment=-1):
    data = np.loadtxt(file_path)

    input_dim = len(data[0])

    if segment == -1:
        segment = input_dim - 1

    inputs = np.asmatrix(data)[:, range(0, segment)]

    outputs = np.asmatrix(data)[:, range(segment, input_dim)]

    return inputs, np.squeeze(np.asarray(outputs))


def error_detect(_xs, _ys, root):
    total_err = 0
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        y = _ys[i]
        res = root.predict(x)
        if not y == res:
            total_err += 1
    return total_err / len(_xs)


def problem_15(x_data, y_data, x_test_data, y_test_data):
    root = fit(x_data, y_data, 1)
    print("=====")
    e_in = error_detect(x_data, y_data, root)
    e_out = error_detect(x_test_data, y_test_data, root)
    print("E_in={i} and E_out = {o}".format(i=e_in, o=e_out))


# x_data, y_data = read_input("hw3_train.dat")
# x_test_data, y_test_data = read_input("hw3_test.dat")
# problem_15(x_data, y_data, x_test_data, y_test_data)
