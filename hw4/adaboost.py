import numpy as np
import matplotlib.pyplot as plt


class DecisionStump:
    def __init__(self, direction=None, threshold=None, feature_index=None):
        self.threshold = threshold
        self.direction = direction
        self.feature_index = feature_index

    def train(self, train_x, train_y, weight_arr=None):

        n_data, n_dim = train_x.shape

        best_error_rate = 1
        best_feedback = []

        for i in range(n_dim):
            feature_vector = train_x[:, i]
            # print(feature_vector)
            theta_arr = [float("-inf")]
            # be careful the parameter 0!
            sorted_train_x = np.sort(feature_vector, 0)
            # print([float(x) for x in sorted_train_x])

            for k in range(n_data - 1):
                theta_arr.append(0.5 * (sorted_train_x[k] + sorted_train_x[k + 1]))

            # print(theta_arr)


            for s in [-1, 1]:
                for theta in theta_arr:
                    ds = DecisionStump(s, theta)
                    err_rate, feedback = ds.calculate_error(feature_vector, train_y, weight_arr)

                    if err_rate < best_error_rate:
                        best_error_rate = err_rate
                        self.threshold = theta
                        self.direction = s
                        self.feature_index = i
                        best_feedback = feedback
        # print(self.threshold, self.direction)
        return best_error_rate, best_feedback

    def calculate_error(self, test_x, test_y, weight_arr=None):

        error_ctr = 0
        len_data = len(test_x)

        answer_feedback = []

        if weight_arr is None:
            weight_arr = np.ones(len_data, dtype=float)

        for i in range(len_data):

            x = float(test_x[i])
            predict_y = self.predict(x)

            correctness = predict_y == int(test_y[i])
            answer_feedback.append(correctness)
            if not correctness:
                error_ctr += weight_arr[i]

        return error_ctr / np.sum(weight_arr), answer_feedback

    def calculate_out_error(self, test_x, test_y, weight_arr=None):

        error_ctr = 0
        len_data = len(test_x)

        answer_feedback = []

        if weight_arr is None:
            weight_arr = np.ones(len_data, dtype=float)

        for i in range(len_data):

            x = float(test_x[i])
            predict_y = self.predict(x)

            correctness = predict_y == int(test_y[i])
            answer_feedback.append(correctness)
            if not correctness:
                error_ctr += weight_arr[i]

        return error_ctr / np.sum(weight_arr)

    def predict(self, _input):

        if isinstance(_input, (int, float, complex)):
            x = _input
        else:
            x = _input[self.feature_index]

        return self.direction if x > self.threshold else -1 * self.direction


class AdaBoost:
    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        pass

    def train(self, train_x, train_y, T, need_e_in=False, need_u_weight=False, need_epsilon=False):
        n_data, n_dim = train_x.shape

        u_weight = np.ones(n_data, dtype=float) / n_data
        alpha_arr = []
        ds_arr = []

        err_rate_arr = []

        if need_u_weight:
            u_weight_sum_arr = []

        if need_epsilon:
            epsilon_arr = []

        if need_e_in:
            e_in_arr = []

        for t in range(T):

            print("======", t, "======")

            # print("sum of u_weight", np.sum(u_weight))

            ds = DecisionStump()
            err_rate, feedback = ds.train(train_x, train_y, u_weight)
            err_rate_arr.append(err_rate)
            ds_arr.append(ds)
            adjust = np.sqrt((1 - err_rate) / err_rate)

            if need_u_weight:
                u_weight_sum_arr.append(np.sum(u_weight))

            # print("direction", ds.direction, "threshold", ds.threshold, "feature_index", ds.feature_index)
            # print("err_rate", err_rate, "adjust", adjust, "alpha", np.log(adjust))
            for k in range(n_data):
                if feedback[k]:
                    u_weight[k] /= adjust
                else:
                    u_weight[k] *= adjust

            if need_epsilon:
                epsilon_arr.append(err_rate)

            if need_e_in:
                e_in_arr.append(1 - (np.sum(1 * feedback) / len(feedback)))

            alpha_arr.append(np.log(adjust))

        self.ds_arr = ds_arr
        self.alpha_arr = alpha_arr

        rtn_arr = []

        if need_e_in:
            rtn_arr.append(e_in_arr)

        if need_u_weight:
            rtn_arr.append(u_weight_sum_arr)

        if need_epsilon:
            rtn_arr.append(epsilon_arr)

        return rtn_arr

    def predict(self, x, max_index=None):

        if max_index is None:
            max_index = len(self.ds_arr)

        predict = 0
        for i in range(max_index):
            ds = self.ds_arr[i]
            alpha = self.alpha_arr[i]
            predict += alpha * ds.predict(x)

        return 1 if predict > 0 else -1


def read_input(file_path, segment=-1):
    data = np.loadtxt(file_path)

    input_dim = len(data[0])

    if segment == -1:
        segment = input_dim - 1

    inputs = np.asmatrix(data)[:, range(0, segment)]

    outputs = np.asmatrix(data)[:, range(segment, input_dim)]

    return inputs, outputs


# print(np.log(np.sqrt(0.76/0.24)))
# exit()

def error_detect(_xs, _ys, ab, max_index=None):
    total_err = 0
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        y = _ys[i]
        res = ab.predict(x, max_index)
        if not y == res:
            total_err += 1
    # print(total_err, len(_xs))
    return total_err / len(_xs)


x_data, y_data = read_input("hw3_train.dat")
y_data = [float(y) for y in y_data]

x_test_data, y_test_data = read_input("hw3_test.dat")
y_test_data = [float(y) for y in y_test_data]


def problem_7(T):
    ab = AdaBoost()
    rtn_arr = ab.train(x_data, y_data, T, need_e_in=True)

    print("E_in(g_1) = {e} and alpha_1 = {a}".format(e=rtn_arr[0][0], a=ab.alpha_arr[0]))
    plt.plot(range(1, T + 1), rtn_arr[0])
    plt.xlabel("t")
    plt.ylabel(r"$E_{in}(g_t)$")
    plt.title(r"$E_{in}(g_t)$ versus t")

    plt.show()



def problem_9(T):
    ab = AdaBoost()
    ab.train(x_data, y_data, T)

    err_in_rate_arr = []
    for t in range(1, T + 1):
        print(t)
        err_in_rate_arr.append(error_detect(x_data, y_data, ab, max_index=t))

    plt.plot(range(1, T + 1), err_in_rate_arr)
    plt.xlabel("t")
    plt.ylabel(r"$E_{in}(G_t)$")
    plt.title(r"$E_{in}(G_t)$ versus t")
    plt.show()


def problem_10(T):
    ab = AdaBoost()
    rtn_arr = ab.train(x_data, y_data, T, need_u_weight=True)

    print("U_2 = {u2} and U_T = {ut} with T = 300".format(u2=rtn_arr[0][1], ut=rtn_arr[0][-1]))

    plt.plot(range(1, T + 1), rtn_arr[0])
    plt.xlabel("t")
    plt.ylabel(r"$U_t$")
    plt.title(r"$U_t$ versus t")

    plt.show()


def problem_11(T):
    ab = AdaBoost()
    rtn_arr = ab.train(x_data, y_data, T, need_epsilon=True)

    print("min_epsilon = {me}".format(me=min(rtn_arr[0])))
    plt.plot(range(1, T + 1), rtn_arr[0])
    plt.xlabel("t")
    plt.ylabel(r"$U_t$")
    plt.title(r"$U_t$ versus t")

    plt.show()


# problem_7(300)



def problem_12(T):
    ab = AdaBoost()
    ab.train(x_data, y_data, T)

    err_rate_arr = []

    ctr = 0

    for ds in ab.ds_arr:
        print(ctr)
        ctr += 1
        feature_index = ds.feature_index
        x_test_vector = x_test_data[:, feature_index]
        # print(x_test_vector)
        # exit()
        err_rate = ds.calculate_out_error(x_test_vector, y_test_data)
        err_rate_arr.append(err_rate)

    print("E_out(g_1) = {me}".format(me=err_rate_arr[0]))

    plt.plot(range(1, T + 1), err_rate_arr)
    plt.xlabel("t")
    plt.ylabel(r"$E_{out}(g_t)$")
    plt.title(r"$E_{out}(g_t)$ versus t")

    plt.show()


def problem_13(T):
    ab = AdaBoost()
    ab.train(x_data, y_data, T)

    print("E_out(G) = {me}".format(me=error_detect(x_test_data, y_test_data, ab)))

    err_out_rate_arr = []
    for t in range(1, T + 1):
        print(t)
        err_out_rate_arr.append(error_detect(x_test_data, y_test_data, ab, max_index=t))


    plt.plot(range(1, T + 1), err_out_rate_arr)
    plt.xlabel("t")
    plt.ylabel(r"$E_{out}(G_t)$")
    plt.title(r"$E_{out}(G_t)$ versus t")
    plt.show()

problem_12(1)