import svmutil as libsvm
import numpy as np
import pylab as plt
from pymlt import mltplot



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
    x = np.transpose(x, (1, 0)).tolist()

    return list(x), list(y)

def test_error_by_input(solution, x, y):

    p_labs, p_acc, p_vals = libsvm.svm_predict(y=y, x=x, m=solution)
    return p_acc


def test_error_by_input_display(solution, x, y):

    p_labs, p_acc, p_vals = libsvm.svm_predict(y=y, x=x, m=solution)

    weight = calculate_weight(solution)
    bias = solution.rho[0]
    print("weight\n", weight)
    print("bias\n", bias)
    print("x", x)
    result = np.dot(np.mat(x), weight) - bias
    print("by_desicision_function", result.tolist())
    print("predict_label", p_labs)
    print("predict_values", p_vals)


    print("----------------------------------------")

    return p_acc


def test_error(solution, test_data_path, flag):
    x, y = load_data(test_data_path, flag)

    p_labs, p_acc, p_vals = libsvm.svm_predict(y=y, x=x, m=solution)

    return 100 - p_acc[0]


def clean_svs(svs):
    svs_list = []
    for sv in svs:
        sv_list = []
        for k in sv:
            if not k < 0:
                sv_list.append(sv[k])

        svs_list.append(sv_list)

    return svs_list


def calculate_weight(_svm_model):
    svs = np.array(clean_svs(list(_svm_model.get_SV())))
    svs_coef = np.array(_svm_model.get_sv_coef())

    weight = np.dot(np.transpose(svs), svs_coef)

    return weight


def train_svm(x, y, config_str):
    prob = libsvm.svm_problem(y, x)
    param = libsvm.svm_parameter(config_str)
    return libsvm.svm_train(prob, param)


def get_weight_length(x, y, config_str):
    m = train_svm(x, y, config_str)
    weight = calculate_weight(m)
    print("The weight is \n {w}".format(w=weight))
    return np.linalg.norm(weight)

    # accuracy_rate = 100 * test(m, "test.txt")
    # print("accuracy_rate: {ar}%".format(ar=accuracy_rate))


def problem_11():
    _x, _y = load_data("train.txt", 0)
    c_exp_list = [-5, -3, -1, 1, 3]
    w_len_list = []
    for c_exp in c_exp_list:
        c = 10 ** c_exp
        config_str = "-q -h 1 -t 0 -c {c}".format(c=c)
        w_len = get_weight_length(_x, _y, config_str)
        w_len_list.append(w_len)

    plt.figure()
    plt.plot(c_exp_list, w_len_list)
    plt.plot(c_exp_list, w_len_list, marker="o", color="red", ls='')
    plt.xlabel("$log_{10}(C)$")
    plt.ylabel("||w||")
    plt.title("problem 11")
    plt.show()


def problem_12():
    _x, _y = load_data("train.txt", 8)
    c_exp_list = [-5, -3, -1, 1, 3]
    e_in_list = []

    for c_exp in c_exp_list:
        c = 10 ** c_exp
        print("With C={c}>>>>>".format(c=c))
        config_str = "-q -h 1 -t 1 -g 1 -r 1 -d 2 -c {c}".format(c=c)
        solution = train_svm(_x, _y, config_str)
        e_in = test_error(solution, "train.txt", 8)
        print(e_in)
        e_in_list.append(e_in)

    plt.figure()
    plt.plot(c_exp_list, e_in_list)
    plt.plot(c_exp_list, e_in_list, marker="o", color="red", ls='')
    plt.xlabel("$log_{10}(C)$")
    plt.ylabel(r'$E_{in}$ (%)')
    plt.title(r'problem 12')
    plt.show()


def problem_13():
    _x, _y = load_data("train.txt", 8)
    c_exp_list = [-5, -3, -1, 1, 3]

    num_sv_list = []
    for c_exp in c_exp_list:
        c = 10 ** c_exp
        config_str = "-q -h 1 -t 1 -g 1 -r 1 -d 2 -c {c}".format(c=c)
        solution = train_svm(_x, _y, config_str)

        num_sv_list.append(len(solution.get_SV()))

    plt.figure()
    plt.plot(c_exp_list, num_sv_list)
    plt.plot(c_exp_list, num_sv_list, marker="o", color="red", ls='')
    plt.xlabel("$log_{10}(C)$")
    plt.ylabel(r'number of support vectors')
    plt.title(r'problem 13')
    plt.show()


def problem_14():
    _x, _y = load_data("train.txt", 0)

    c_exp_list = [-3, -2, -1, 0, 1]
    distance_list = []

    def kernel(x_1, x_2, gamma=80):
        return np.exp(-1 * gamma * (np.linalg.norm(x_1 - x_2) ** 2))

    for c_exp in c_exp_list:

        c = 10 ** c_exp
        print("With C={c}>>>>>".format(c=c))
        config_str = "-q -t 2 -g 80 -c {c}".format(c=c)

        solution = train_svm(_x, _y, config_str)
        svs_dict = solution.get_SV()
        sv_coef = np.array(solution.get_sv_coef())

        max_range = len(svs_dict)
        tmp = 0

        print(max_range)

        for i_1 in range(max_range):
            for i_2 in range(max_range):
                sv_dict_1 = svs_dict[i_1]
                sv_dict_2 = svs_dict[i_2]
                x_1 = np.array([[sv_dict_1[1]], [sv_dict_1[2]]])
                x_2 = np.array([[sv_dict_2[1]], [sv_dict_2[2]]])

                tmp += sv_coef[i_1] * sv_coef[i_2] * kernel(x_1, x_2)

            if i_1 % 100 == 0:
                print(i_1)
        print(tmp)
        distance = 1 / np.sqrt(tmp)
        distance_list.append(distance)

    plt.figure()
    plt.plot(c_exp_list, distance_list)
    plt.plot(c_exp_list, distance_list, marker="o", color="red", ls='')
    plt.xlabel("$log_{10}(C)$")
    plt.ylabel("distance of free support vector to the hyperplane in Z-space")
    plt.title('problem 14')
    plt.show()


def problem_15():
    flag = 0
    _x, _y = load_data("train.txt", flag)

    gamma_exp_list = [0, 1, 2, 3, 4]
    e_out_list = []

    for gamma_exp in gamma_exp_list:
        gamma = 10 ** gamma_exp
        print("With gamma={g}>>>>>".format(g=gamma))
        config_str = "-q -g {g} -c 0.1".format(g=gamma)
        solution = train_svm(_x, _y, config_str)
        e_out = test_error(solution, "test.txt", flag)
        e_out_list.append(e_out)

    plt.figure()
    plt.plot(gamma_exp_list, e_out_list)
    plt.plot(gamma_exp_list, e_out_list, marker="o", color="red", ls='')
    plt.xlabel("$log_{10}(\gamma$)")
    plt.ylabel("$E_{out}$(%)")
    plt.title('problem 15')
    plt.show()

def validation(_x, _y, size=1000):
    random_indices = np.random.choice(len(_x), size, replace=False)

    _x_test = [_x[i] for i in random_indices]
    _y_test = [_y[i] for i in random_indices]

    _x_train = [_x[i] for i in range(len(_x)) if i not in random_indices]
    _y_train = [_y[i] for i in range(len(_x)) if i not in random_indices]

    return _x_train, _y_train, _x_test, _y_test


def problem_16():
    flag = 0
    _x, _y = load_data("train.txt", flag)

    gamma_exp_list = [-1, 0, 1, 2, 3]
    gamma_counter = dict()

    for exp in gamma_exp_list:
        gamma_counter[exp] = 0

    for _ in range(100):

        _x_train, _y_train, _x_test, _y_test = validation(_x, _y, 1000)

        max_index = -100
        max_accuracy = -100

        for gamma_exp in gamma_exp_list:
            gamma = 10 ** gamma_exp
            config_str = "-q -t 2 -c 0.1 -g {g} ".format(g=gamma)
            solution = train_svm(_x_train, _y_train, config_str)
            acc = 100 - test_error_by_input(solution, _x_test, _y_test)

            if acc > max_accuracy:
                max_index = gamma_exp
                max_accuracy = acc

            del solution

        print(_)

        gamma_counter[max_index] += 1

    fig, ax = plt.subplots()
    gamma_labels = []
    gamma_performance = []
    for gamma in gamma_exp_list:
        gamma_labels.append('$log_{10}(\gamma)$=' + str(gamma))
        gamma_performance.append(gamma_counter[gamma])

    gamma_labels = tuple(gamma_labels)
    gamma_performance = tuple(gamma_performance)
    y_pos = np.arange(len(gamma_labels))

    ax.barh(y_pos, gamma_performance, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(gamma_labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Number of selected times')
    ax.set_title('problem 16')

    for i, v in enumerate(gamma_performance):
        ax.text(v, i, str(v), color='blue', fontweight='bold')

    plt.show()



problem_11()
problem_12()
problem_13()
problem_14()
problem_15()
problem_16()
