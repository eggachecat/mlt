import svmutil
import numpy as np


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

def get_01_error(y_arr, predict_arr):

    total_err = 0
    for i in range(len(y_arr)):
        _y = y_arr[i]
        predict = np.sign(predict_arr[i])
        total_err += 1 * (predict != _y)

    return total_err


x, y = read_input("hw2_lssvm_all.dat")
train_size = 400
train_x = x[0:train_size]
train_y = y[0:train_size]
test_x = x[train_size:len(x)]
test_y = y[train_size:len(y)]



train_x = train_x.tolist()
train_y = [float(i) for sub in train_y for i in sub]
test_x = test_x.tolist()
test_y = [float(i) for sub in test_y.tolist() for i in sub]


gamma_set = [32, 2, 0.125]
c_set = [0.001, 1, 1000]

for _gamma in gamma_set:
    for _c in c_set:
        print("With gamma = {g} and C = {_c} ".format(g=_gamma, _c=_c))
        prob = svmutil.svm_problem(train_y, train_x)
        param = svmutil.svm_parameter('-s 3 -c {_c} -e 0.5 -t 2 -g {_g} -q'.format(_c=_c, _g=_gamma))
        m = svmutil.svm_train(prob, param)

        ein_labs, ein_acc, ein_vals = svmutil.svm_predict(y=train_y, x=train_x, m=m)
        eout_labs, eout_acc, eout_vals = svmutil.svm_predict(y=test_y, x=test_x, m=m)

        print(
            "\t We have (0/1)-error: E_in(g)={e_in}% and E_out(g)={e_out}%".format(
                e_in=100 * get_01_error(train_y, ein_labs) / len(train_x),
                e_out=100 * get_01_error(test_y, eout_labs) / len(test_x)))
