


import svmutil as libsvm
import numpy as np
from pymlt import mltplot
from sklearn.svm import SVC

def draw_data(data_path, flag):
    data_set = np.loadtxt(data_path)
    for row in data_set:
        row[0] = 1 if row[0] == flag else 0

    mltplot.draw_data(data_set, True, 0)


def load_data(data_path, flag):

    data_set = np.loadtxt(data_path)
    data_set = np.transpose(data_set, (1, 0))

    y = data_set[0]
    y = 2 * (y == flag) - 1

    x = np.delete(data_set, 0, 0)
    x = np.transpose(x, (1, 0))

    return x.tolist(), y.tolist()


def test(solution, test_data_path):

    x, y = load_data(test_data_path, 0)

    test_correct_times = 0
    test_total_times = len(y)

    for i in range(test_total_times):
        ans = y[i]
        if ans == solution.predict([x[i]]):
            test_correct_times += 1

    return test_correct_times / test_total_times;


def get_coef(x, y, c):

    clf = SVC(C=c, kernel='linear', shrinking=False)
    clf.fit(x, y)
    print(clf.coef_)
    print(np.linalg.norm(clf.coef_))
    # accuracy_rate = 100 * test(clf, "test.txt")
    # print("accuracy_rate: {ar}%".format(ar=accuracy_rate))

_x, _y = load_data("train.txt", 0)
c_set = [10 ** -5, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 1]
for c in c_set:
    print("{c}>>>>>".format(c=c))
    get_coef(_x, _y, c)

