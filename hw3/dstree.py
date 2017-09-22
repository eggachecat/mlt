import numpy as np
from pymlt import mltplot


class DecisionStump:
    def __init__(self, direction, threshold):
        self.threshold = threshold
        self.direction = direction

    def split_data(self, x_data):

        left_indices = []
        right_indices = []

        for i in range(len(x_data)):
            if self.predict(x_data[i]) < 0:
                left_indices.append(i)
            else:
                right_indices.append(i)

        return left_indices, right_indices

    def predict(self, x):
        return self.direction if x > self.threshold else -1 * self.direction


class BranchNode:
    _id = 0

    def __init__(self, ds, fi):
        self.left = None
        self.right = None
        self.ds = ds
        self.fi = fi
        self.id = BranchNode._id
        BranchNode._id += 1

    def predict(self, x):

        if self.ds.predict(x[self.fi]) < 0:
            if isinstance(self.left, BranchNode):
                return self.left.predict(x)
            else:
                return self.left
        else:
            if isinstance(self.right, BranchNode):
                return self.right.predict(x)
            else:
                return self.right

    def punching_leave_predict(self, x, leaf_id, is_left_leaf):

        # print(self.id, leaf_id)
        if self.id == leaf_id:
            if is_left_leaf:
                if isinstance(self.left, BranchNode):
                    return self.left.punching_leave_predict(x, leaf_id, is_left_leaf)
                else:
                    return self.left
            else:
                if isinstance(self.right, BranchNode):
                    return self.right.punching_leave_predict(x, leaf_id, is_left_leaf)
                else:
                    return self.right
        else:
            if self.ds.predict(x[self.fi]) < 0:
                if isinstance(self.left, BranchNode):
                    return self.left.punching_leave_predict(x, leaf_id, is_left_leaf)
                else:
                    return self.left
            else:
                if isinstance(self.right, BranchNode):
                    return self.right.punching_leave_predict(x, leaf_id, is_left_leaf)
                else:
                    return self.right


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = 10
        self.branch_node_arr = []

    def calculate_e_in(self, ds, test_x, test_y):

        error_ctr = 0
        len_data = len(test_x)
        for i in range(len_data):
            x = float(test_x[i])
            predict_y = ds.predict(x)
            if not predict_y == test_y[i]:
                error_ctr += 1

        return error_ctr / len_data

    @staticmethod
    def calculate_gini(ds, test_y):

        correct_ctr = {}
        len_data = len(test_y)
        for c in np.unique(test_y):
            correct_ctr[c] = 0

        for y in test_y:
            correct_ctr[y] += 1

        gini = 0
        for c in correct_ctr:
            gini += (correct_ctr[c] / len_data) ** 2
        gini = 1 - gini

        return gini

    def train(self, train_x, train_y, free_depth):

        print("--------------------")

        n_data, n_dim = train_x.shape

        if free_depth == 0 or np.sum(np.asmatrix(train_y) == train_y[0]) == len(train_y) or np.sum(
                (train_x == train_x[0, :]).all(axis=1)) == len(train_x):
            unique, counts = np.unique(train_y, return_counts=True)
            sorted_counts_indices = np.argsort(counts)
            return unique[sorted_counts_indices[-1]]

        best_ds = None
        best_impurity = float("inf")
        best_index = -1

        for i in range(n_dim):

            feature_vector = train_x[:, i]

            sorted_feature_vector = np.sort(feature_vector, 0)

            theta_arr = (np.r_[sorted_feature_vector[0:1, :] - 0.1, sorted_feature_vector] + np.r_[
                sorted_feature_vector, sorted_feature_vector[-1:, :] + 0.1]) / 2

            for s in [1]:
                for theta in theta_arr:
                    ds = DecisionStump(s, theta)

                    l_is, r_is = ds.split_data(feature_vector)
                    l_y = train_y[l_is]
                    r_y = train_y[r_is]

                    impurity = len(l_y) * DecisionTree.calculate_gini(ds, l_y) + len(r_y) * DecisionTree.calculate_gini(
                        ds, r_y)

                    if impurity < best_impurity:
                        best_impurity = impurity
                        best_ds = ds
                        best_index = i

        print("best_theta is", best_ds.threshold)
        print("best_gini is", best_impurity)
        print("best_index is", best_index)

        best_feature_vector = train_x[:, best_index]

        l_is, r_is = best_ds.split_data(best_feature_vector)

        left_x_data = train_x[l_is]
        left_y_data = train_y[l_is]

        right_x_data = train_x[r_is]
        right_y_data = train_y[r_is]

        print("len(left_y_data)", len(left_y_data))
        print("len(right_y_data)", len(right_y_data))

        bn = BranchNode(best_ds, best_index)

        bn.left = self.train(np.asmatrix(left_x_data), left_y_data, free_depth - 1)
        bn.right = self.train(np.asmatrix(right_x_data), right_y_data, free_depth - 1)

        return bn


def read_input(file_path, segment=-1):
    data = np.loadtxt(file_path)

    input_dim = len(data[0])

    if segment == -1:
        segment = input_dim - 1

    inputs = np.asmatrix(data)[:, range(0, segment)]

    outputs = np.asmatrix(data)[:, range(segment, input_dim)]

    return inputs, outputs


def LMR(node, depth=0):
    print("-------{depth}-start--------".format(depth=depth))

    print("This node id :", node.id)
    print("This direction", node.ds.direction)
    print("This threshold", node.ds.threshold)
    print("This index", node.fi)
    print("----")

    if isinstance(node.left, BranchNode):
        print("Left node id :", node.left.id)
    else:
        print("Left-Leaf: {val}".format(val=node.left))

    if isinstance(node.right, BranchNode):
        print("Right node id :", node.right.id)
    else:
        print("Right-Leaf: {val}".format(val=node.right))

    if isinstance(node.left, BranchNode):
        LMR(node.left, depth + 1)

    if isinstance(node.right, BranchNode):
        LMR(node.right, depth + 1)

    print("-------{depth}-done--------".format(depth=depth))


def error_detect(_xs, _ys, root):
    total_err = 0
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        y = _ys[i]
        res = root.predict(x)
        if not y == res:
            total_err += 1
    return total_err / len(_xs)


def punching_leaf_error_detect(_xs, _ys, root, node_id, is_left_leaf):
    total_err = 0
    for i in range(len(_xs)):
        x = _xs[i, :].tolist()[0]
        y = _ys[i]
        res = root.punching_leave_predict(x, node_id, is_left_leaf)
        if not y == res:
            total_err += 1
    return total_err / len(_xs)


x_data, y_data = read_input("hw3_train.dat")
y_data = np.squeeze(np.asarray(y_data))

x_test_data, y_test_data = read_input("hw3_test.dat")
y_test_data = np.squeeze(np.asarray(y_test_data))


def problem_14(x_data, y_data, x_test_data, y_test_data):
    dt = DecisionTree(1)
    root = dt.train(x_data, y_data, 1000)
    LMR(root, 0)


def problem_15(x_data, y_data, x_test_data, y_test_data):
    dt = DecisionTree(1)
    root = dt.train(x_data, y_data, 1)
    print("=====")
    e_in = error_detect(x_data, y_data, root)
    e_out = error_detect(x_test_data, y_test_data, root)
    print("E_in={i} and E_out = {o}".format(i=e_in, o=e_out))


def problem_16(x_data, y_data, x_test_data, y_test_data):
    dt = DecisionTree(1)
    root = dt.train(x_data, y_data, 1000)

    for node_id in [2, 5, 6, 7, 8, 9]:
        for is_left in [True, False]:
            print("=======")
            print("Punching {il} leaf of branch id={n_id}".format(n_id=node_id, il="left" if is_left else "right"))
            e_in = punching_leaf_error_detect(x_data, y_data, root, node_id, is_left)
            e_out = punching_leaf_error_detect(x_test_data, y_test_data, root, node_id, is_left)
            print("E_in={i} and E_out = {o}".format(i=e_in, o=e_out))


# problem_14(x_data, y_data, x_test_data, y_test_data)
problem_15(x_data, y_data, x_test_data, y_test_data)
# problem_16(x_data, y_data, x_test_data, y_test_data)
