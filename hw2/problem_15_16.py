import numpy as np


class LinearLSSVM:
    def __init__(self):
        self.weight = None

    def train(self, x_matrix, y_matrix, _lambda):
        n = x_matrix.shape[1]
        self.weight = np.dot(
            np.dot(np.linalg.inv(_lambda * np.identity(n, dtype=float) + np.dot(np.transpose(x_matrix), x_matrix)),
                   np.transpose(x_matrix)),
            y_matrix)

    def predict(self, _x):
        return np.dot(np.transpose(self.weight), _x)


def read_input(file_path, segment=-1):
    data = np.loadtxt(file_path)

    input_dim = len(data[0])

    if segment == -1:
        segment = input_dim - 1

    inputs = np.asmatrix(data)[:, range(0, segment)]

    # for bias
    inputs = np.c_[np.ones(len(inputs)), inputs]
    outputs = np.asmatrix(data)[:, range(segment, input_dim)]

    return inputs, outputs


def request_data(x_pool, y_pool, size):
    x_pool = x_pool.tolist()
    y_pool = list(y_pool)

    idx_list = np.random.randint(len(x_pool), size=size)

    return np.matrix([x_pool[idx] for idx in idx_list], dtype=float), np.transpose(
        np.matrix([float(y_pool[idx]) for idx in idx_list]))


def value_bag(x_arr, y_arr, bags):
    total_err = 0

    for i in range(len(x_arr)):
        _x = np.transpose(x_arr[i])
        _y = y_arr[i]

        consensus = 0
        for g in bags:
            consensus += g.predict(_x)

        total_err += 1 * (consensus * _y < 0)

    return total_err


x, y = read_input("hw2_lssvm_all.dat")
train_size = 400
train_x = x[0:train_size]
train_y = y[0:train_size]
test_x = x[train_size:len(x)]
test_y = y[train_size:len(y)]


print(np.sum(train_y>0))

MAX_BAG_TIME = 201
BAG_DATA_SIZE = 400

lambda_set = [0, 0.01, 0.1, 1, 10, 100]

for _lambda in lambda_set:
    print("With lambda = {l}  ({itt} iterations for bagging and {ds}-data-size)".format(l=_lambda, itt=MAX_BAG_TIME,
                                                                                        ds=BAG_DATA_SIZE))
    g_arr = []
    for _ in range(MAX_BAG_TIME):
        x_data, y_data = request_data(train_x, train_y, BAG_DATA_SIZE)

        svm = LinearLSSVM()
        svm.train(x_data, y_data, _lambda)
        g_arr.append(svm)

    e_in = value_bag(train_x, train_y, g_arr)
    e_out = value_bag(test_x, test_y, g_arr)
    print(
        "\t We have (0/1)-error: E_in(g)={e_in}% and E_out(g)={e_out}%".format(
            e_in=100 * float(e_in) / len(train_x),
            e_out=100 * float(e_out) / len(test_x)))
