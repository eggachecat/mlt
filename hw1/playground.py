import os
import random
import math

from svmutil import *

import numpy as np



def validation(_x, _y, size=1000):

    random_indices = np.random.choice(len(_x), size, replace=False)

    _x_test = [_x[i] for i in random_indices]
    _y_test = [_y[i] for i in random_indices]

    _x_train = [_x[i] for i in range(len(_x)) if i not in random_indices]
    _y_train = [_y[i] for i in range(len(_x)) if i not in random_indices]

    return _x_train, _y_train, _x_test, _y_test

def load_data(data_path, flag):
    data_set = np.loadtxt(data_path)
    data_set = np.transpose(data_set, (1, 0))

    y = data_set[0]
    y = 2 * (y == flag) - 1

    x = np.delete(data_set, 0, 0)
    x = np.transpose(x, (1, 0)).tolist()

    return x, y


trainfile = 'train.txt'
gamaArr = [0, 0, 0, 0, 0]
for time in range(0, 2):
    acc = 0
    n_acc = 0
    print(str(time) + "！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
    for i in range(0, 5):
        parameter = '-q -t 2 -c 0.1 -g ' + str(math.pow(10, i))
        x, y = load_data(trainfile, 0)
        # x = list(x)
        # y = list(y)
        # yt = list()
        # xt = list()
        # for j in range(0, 1000):
        #     index = random.randint(0, len(x) - 1)
        #     yt.append(y[index])
        #     xt.append(x[index])
        #     del y[index]
        #     del x[index]

        x, y, xt, yt = validation(x, y)
        model = svm_train(y, x, parameter)
        p_label, p_acc, p_val = svm_predict(yt, xt, model)
        if (p_acc[0] > acc):
            acc = p_acc[0]
            n_acc = i
        print(parameter)
        print(p_acc[0])
        print(acc)
        print(n_acc)
    gamaArr[n_acc] = gamaArr[n_acc] + 1
    print(gamaArr)
