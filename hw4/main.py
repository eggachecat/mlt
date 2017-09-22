import dstree
from sklearn.utils import resample
import time
import numpy as np
import mltplot
from sklearn.tree import DecisionTreeClassifier

import sys


# my alg is so slow...
# that 10-trees will take about 3-seconds
# which implies it will take 9000s to complete 30000 trees

def calculate_e_in(y_truth, y_pred):
    n_data = len(y_pred)
    cor = np.sum(y_pred == y_truth)
    return 1 - cor / n_data


def error_detect(_xs, _ys, root):
    total_err = 0
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        y = _ys[i]
        res = root.predict(x)
        if not y == res:
            total_err += 1
    return total_err / len(_xs)


def get_predict(_xs, root):
    res_arr = []
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        res_arr.append(root.predict(x))
    return res_arr


def error_detect_forest(_ys, pred_arr):
    n_data = len(_ys)
    pred = np.sign(np.sum(pred_arr, axis=0))

    return 1 - np.sum(pred == _ys) / n_data


def problem_12(X, y, times=30):
    e_in_arr = []
    occ_arr = []
    idx_ctr = 0

    for i in range(times):
        b_X, b_y = resample(X, y)
        root = dstree.fit(b_X, b_y, 100)

        e_in = error_detect(X, y, root)
        if e_in not in e_in_arr:
            e_in_arr.append(e_in)
            occ_arr.append(0)
            idx_ctr += 1
        occ_arr[e_in_arr.index(e_in)] += 1

    canvas = mltplot.MltCanvas()
    ax = canvas.add_canvas(1)
    ax.vlines(e_in_arr, [0], occ_arr)
    ax.set_xticks(e_in_arr)
    ax.set_ylim(ymin=0)
    canvas.set_x_label("$E_{in}(g_t)$", sub_canvas_id=1)
    canvas.set_y_label("frequency (x10)", sub_canvas_id=1)
    canvas.set_title("$E_{in}(g_t)$ VS frequency", sub_canvas_id=1)
    canvas.save("output/12.png")
    canvas.froze()


def problem_13(X, y, times=30):
    e_in_arr = []
    root_arr = []

    for i in range(times):
        b_X, b_y = resample(X, y)
        root = dstree.fit(b_X, b_y, 100)
        root_arr.append(root)

    predict_arr = []

    for _root in root_arr:
        predict_arr.append(get_predict(X, _root))

    predict_arr = np.matrix(predict_arr)

    for t in range(1, 1 + times):
        e_in = error_detect_forest(y, predict_arr[:t])
        e_in_arr.append(e_in)

    canvas = mltplot.MltCanvas()
    canvas.draw_line_chart_2d(range(1, 1 + times), e_in_arr, line_style="solid")
    canvas.set_x_label("t", sub_canvas_id=1)
    canvas.set_y_label("$E_{in}(G_t)$", sub_canvas_id=1)
    canvas.set_title("$E_{in}(G_t)$ VS t", sub_canvas_id=1)
    canvas.save("output/13.png")
    canvas.froze()


def problem_14(X, y, X_test, y_test, times=30):
    e_in_arr = []
    root_arr = []

    for i in range(times):
        b_X, b_y = resample(X, y)
        root = dstree.fit(b_X, b_y, 100)
        root_arr.append(root)

    predict_arr = []

    for _root in root_arr:
        predict_arr.append(get_predict(X_test, _root))

    predict_arr = np.matrix(predict_arr)

    for t in range(1, 1 + times):
        e_in = error_detect_forest(y_test, predict_arr[:t])
        e_in_arr.append(e_in)

    canvas = mltplot.MltCanvas()
    canvas.draw_line_chart_2d(range(1, 1 + times), e_in_arr, line_style="solid")
    canvas.set_x_label("t", sub_canvas_id=1)
    canvas.set_y_label("$E_{out}(G_t)$", sub_canvas_id=1)
    canvas.set_title("$E_{out}(G_t)$ VS t", sub_canvas_id=1)
    canvas.save("output/14.png")
    canvas.froze()


def problem_15(X, y, times=30):
    e_in_arr = []
    root_arr = []

    for i in range(times):
        b_X, b_y = resample(X, y)
        root = dstree.fit(b_X, b_y, 1)
        root_arr.append(root)

    predict_arr = []

    for _root in root_arr:
        predict_arr.append(get_predict(X, _root))

    predict_arr = np.matrix(predict_arr)

    for t in range(1, 1 + times):
        e_in = error_detect_forest(y, predict_arr[:t])
        e_in_arr.append(e_in)

    canvas = mltplot.MltCanvas()
    canvas.draw_line_chart_2d(range(1, 1 + times), e_in_arr, line_style="solid")
    canvas.set_x_label("t", sub_canvas_id=1)
    canvas.set_y_label("$E_{in}(G_t)$", sub_canvas_id=1)
    canvas.set_title("$E_{in}(G_t)$ VS t with one branch", sub_canvas_id=1)
    canvas.save("output/15.png")
    canvas.froze()


def problem_16(X, y, X_test, y_test, times=30):
    e_in_arr = []
    root_arr = []

    for i in range(times):
        b_X, b_y = resample(X, y)
        root = dstree.fit(b_X, b_y, 1)
        root_arr.append(root)

    predict_arr = []

    for _root in root_arr:
        predict_arr.append(get_predict(X_test, _root))

    predict_arr = np.matrix(predict_arr)

    for t in range(1, 1 + times):
        e_in = error_detect_forest(y_test, predict_arr[:t])
        e_in_arr.append(e_in)

    canvas = mltplot.MltCanvas()
    canvas.draw_line_chart_2d(range(1, 1 + times), e_in_arr, line_style="solid")
    canvas.set_x_label("t", sub_canvas_id=1)
    canvas.set_y_label("$E_{out}(G_t)$", sub_canvas_id=1)
    canvas.set_title("$E_{out}(G_t)$ VS t with one branch", sub_canvas_id=1)
    canvas.save("output/16.png")
    canvas.froze()


from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


def get_e_in(y_true, y_pred):
    n_data = len(y_true)
    return 1 - np.sum(y_true == y_pred) / n_data


import pylab as plt


def sk_problem_12(X, y, times):
    e_in_arr = []
    occ_arr = []
    idx_ctr = 0

    # clf = RandomForestClassifier(max_features=None, n_estimators=times)
    # clf.fit(train_X, train_y)
    n_data = len(y)
    for i in range(times):
        b_i = np.random.choice(n_data, n_data)
        b_X, b_y = X[b_i], y[b_i]

        clf_ = tree.DecisionTreeClassifier()
        clf_.fit(b_X, b_y)

        e_in = get_e_in(y, clf_.predict(X))
        if e_in not in e_in_arr:
            e_in_arr.append(e_in)
            occ_arr.append(0)
            idx_ctr += 1
        occ_arr[e_in_arr.index(e_in)] += 1

    #

    #
    # print(np.mean(e_in_arr))

    canvas = mltplot.MltCanvas()
    ax = canvas.add_canvas(1)
    ax.vlines(e_in_arr, [0], occ_arr)
    ax.set_xticks(e_in_arr)
    ax.set_ylim(ymin=0)
    canvas.set_x_label("$E_{in}(g_t)$", sub_canvas_id=1)
    canvas.set_y_label("frequency (x10)", sub_canvas_id=1)
    canvas.set_title("$E_{in}(g_t)$ VS frequency", sub_canvas_id=1)
    canvas.save("output/sklearn_12.png")
    canvas.froze()


if __name__ == "__main__":

    train_X, train_y = dstree.read_input("hw3_train.dat")
    test_X, test_y = dstree.read_input("hw3_test.dat")
    problem_12(train_X, train_y, 30000)

    # sk_problem_12(train_X, train_y, 1000)
    #
    # p = sys.argv[1]
    # if p == "12":
    #     problem_12(train_X, train_y, 30000)
    # if p == "13":
    #     problem_13(train_X, train_y, 30000)
    # if p == "14":
    #     problem_14(train_X, train_y, test_X, test_y, 30000)
    # if p == "15":
    #     problem_15(train_X, train_y, 30000)
    # if p == "16":
    #     problem_16(train_X, train_y, test_X, test_y, 30000)


    #
    # s = time.time()
    # problem_12(train_X, train_y, 3000)
    # problem_13(train_X, train_y, 3000)
    # problem_14(train_X, train_y, test_X, test_y, 3000)
    # problem_15(train_X, train_y, 3000)
    # problem_16(train_X, train_y, test_X, test_y, 3000)
    # e = time.time()
    # print(e - s)


# clf = DecisionTreeClassifier(random_state=0)
#     clf.fit(b_X, b_y)
#     y_truth, y_pred = y, clf.predict(X)
#     e_in = calculate_e_in(y_truth, y_pred)
#
#     if e_in not in e_in_arr_2:
#         e_in_arr_2.append(e_in)
#         occ_arr_2.append(0)
#         idx_ctr_2 += 1
#     occ_arr_2[e_in_arr_2.index(e_in) - 1] += 1
# e = time.time()
# print(e - s)
# canvas = mltplot.MltCanvas(shape=(2, 1))



# ax_2 = canvas.add_canvas(2)
# ax_2.vlines(e_in_arr_2, [0], occ_arr_2)
# ax_2.set_xticks(e_in_arr_2)
# ax_2.set_ylim(ymin=0)
# canvas.set_x_label("$E_{in}(g_t)$", sub_canvas_id=2)
# canvas.set_y_label("frequency", sub_canvas_id=2)
# canvas.set_title("$E_in(g_t)$ VS frequency", sub_canvas_id=2)
#
# canvas.froze()

# e = time.time()
# print(e - s)
# x_test_data, y_test_data = dstree.read_input("hw3_test.dat")

# so let me use sklearn..

# def calculate_e_in(y_truth, y_pred):
#     n_data = len(y_pred)
#     cor = np.sum(y_pred == y_truth)
#     return 1 - cor / n_data
#
#
# X, y = dstree.read_input("hw3_train.dat")
#
#
# def problem_12(X, y):


# for i in range(300):
#     b_X, b_y = resample(X, y)
#
#     clf = DecisionTreeClassifier(random_state=0)
#     clf.fit(b_X, b_y)
#     y_truth, y_pred = y, clf.predict(X)
#     e_in = calculate_e_in(y_truth, y_pred)
#
#     if e_in not in e_in_arr_2:
#         e_in_arr_2.append(e_in)
#         occ_arr_2.append(0)
#         idx_ctr_2 += 1
#     occ_arr[e_in_arr_2.index(e_in) - 1] += 1

# avg = np.sum(np.array(e_in_arr) * np.array(occ_arr)) / np.sum(np.array(occ_arr))
# print(avg)



# problem_12(X, y)
