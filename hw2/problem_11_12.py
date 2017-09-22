import numpy as np


class LSSVM:
    def __init__(self, kernel_function):
        self.beta = None
        self.svs = None
        self.svs_len = 0
        self.kernel_function = kernel_function

    def train(self, x_arr, y_arr, _lambda):
        n = len(x_arr)

        # print(self.compute_kernel_matrix(x_arr))
        # exit()
        self.beta = np.dot(np.linalg.inv(_lambda * np.identity(n, dtype=float) + self.compute_kernel_matrix(x_arr)),
                           y_arr)

        # print(self.beta)
        self.svs = x_arr
        self.svs_len = len(x_arr)

    def test(self, x_arr, y_arr):

        total_err = 0

        for i in range(len(x_arr)):
            _y = y_arr[i]
            _x = x_arr[i]
            predict = 0

            for n in range(self.svs_len):
                predict += self.beta[n] * self.kernel_function(self.svs[n], _x)

            # calculate as 0/1 error
            predict = np.sign(predict)
            total_err += 1 * (predict != _y)

        return float(total_err)

    def compute_kernel_matrix(self, _x):
        n = len(_x)
        km = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                km[i, j] = self.kernel_function(_x[i], _x[j])

        return km


def gaussian_RBF(_gamma):
    def kernel_function(v_x_1, v_x_2):
        norm = np.linalg.norm(v_x_1 - v_x_2)
        return np.exp(-1 * _gamma * norm * norm)

    return kernel_function


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


x, y = read_input("hw2_lssvm_all.dat")

train_size = 400
train_x = x[0:train_size]

train_y = y[0:train_size]
test_x = x[train_size:len(x)]
test_y = y[train_size:len(y)]

gamma_set = [32, 2, 0.125]
lambda_set = [0.001, 1, 1000]

for _gamma in gamma_set:
    for _lambda in lambda_set:
        print("With gamma = {g} and lambda = {l} ".format(g=_gamma, l=_lambda))

        svm = LSSVM(gaussian_RBF(_gamma))
        svm.train(train_x, train_y, _lambda)
        e_in = svm.test(train_x, train_y)
        e_out = svm.test(test_x, test_y)

        print(
            "\t We have (0/1)-error: E_in(g)={e_in}% and E_out(g)={e_out}%".format(
                e_in=100 * e_in / len(train_x),
                e_out=100 * e_out / len(test_x)))
