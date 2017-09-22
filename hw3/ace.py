# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

# 为了实现决策树，需建立树结点的类
# 定义树结点
class Node:
    def __init__(self, theta, index, value=None):
        self.theta = theta  # 划分的阈值
        self.index = index  # 选用的维度
        self.value = value  # 根节点的值
        self.leftNode = None
        self.rightNode = None


# In[6]:

# 定义Gini系数---作为每个子集“好坏”的衡量标准
def gini(Y):
    l = Y.shape[0]
    if l == 0:
        return 1
    print()
    return 1 - (np.sum(Y == 1) / l) ** 2 - (np.sum(Y == -1) / l) ** 2


# In[7]:

# 为了便于实现，找出每一维度下的最佳划分阈值和对应的branch值 --- 但这样实现代价是运行速度
# 单维情况下的最佳树桩---大于等于为1类
def one_stump(X, Y, thres):
    l = thres.shape[0]
    mini = Y.shape[0]
    for i in range(l):
        Y1 = Y[X < thres[i]]
        Y2 = Y[X >= thres[i]]
        print(len(Y1), len(Y2))
        judge = Y1.shape[0] * gini(Y1) + Y2.shape[0] * gini(Y2)
        print(thres[i], gini(Y1), gini(Y2), judge)
        exit()
        if mini > judge:
            mini = judge;
            b = thres[i]


    return mini, b


# In[8]:

# 找出最佳划分的阈值和对应的维度
# 结合全部维数的决策树桩
def decision_stump(X, Y):
    row, col = X.shape
    Xsort = np.sort(X, 0)
    thres = (np.r_[Xsort[0:1, :] - 0.1, Xsort] + np.r_[Xsort, Xsort[-1:, :] + 0.1]) / 2

    # print(thres)
    # exit()
    mpurity = row;
    mb = 0;
    index = 0
    for i in range(col):
        purity, b = one_stump(X[:, i], Y[:, 0], thres[:, i])

        print(purity, b)

        if mpurity > purity:
            mpurity = purity;
            mb = b;
            index = i
    return mb, index, mpurity


# In[9]:

# 定义划分终止的条件
# 终止条件
def stop_cond(X, Y):
    if np.sum(Y != Y[0]) == 0 or X.shape[0] == 1 or np.sum(X != X[0, :]) == 0:
        return True
    return False


# In[10]:

# 定义完全生长的决策树
def dTree(X, Y, depth=0):

    print(depth, len(Y))


    if stop_cond(X, Y):
        node = Node(None, None, Y[0])
        return node
    b, index, gini = decision_stump(X, Y)


    print("best_theta is", b)
    print("best_gini is", gini)
    print("best_index is", index)
    pos1 = X[:, index] < b
    pos2 = X[:, index] >= b
    leftX = X[pos1, :]
    leftY = Y[pos1, 0:1]
    rightX = X[pos2, :]
    rightY = Y[pos2, 0:1]
    node = Node(b, index)

    print("len(left_y_data)", len(leftY))
    print("len(right_y_data)", len(rightY))
    print("--------------------")
    node.leftNode = dTree(leftX, leftY, depth + 1)
    node.rightNode = dTree(rightX, rightY, depth + 1)
    return node


# In[11]:

# 定义只进行一次划分的决策树（夸张的剪枝）
def dTree_one(X, Y):
    b, index = decision_stump(X, Y)
    pos1 = X[:, index] < b;
    pos2 = X[:, index] >= b
    node = Node(b, index)
    value1 = 1 if np.sign(np.sum(Y[pos1])) >= 0 else -1
    value2 = 1 if np.sign(np.sum(Y[pos2])) >= 0 else -1
    node.leftNode = Node(None, None, np.array([value1]))
    node.rightNode = Node(None, None, np.array([value2]))
    return node


# In[12]:

# 预测函数---基于决策树对单个样本进行的预测
def predict_one(node, X):
    if node.value is not None:
        return node.value[0]
    thre = node.theta;
    index = node.index
    if X[index] < thre:
        return predict_one(node.leftNode, X)
    else:
        return predict_one(node.rightNode, X)


# In[13]:

# 基于决策树的预测结果及其错误率衡量函数
def err_fun(X, Y, node):
    row, col = X.shape
    Yhat = np.zeros(Y.shape)
    for i in range(row):
        Yhat[i] = predict_one(node, X[i, :])
    return Yhat, np.sum(Yhat != Y) / row


# In[26]:

# bagging函数
def bagging(X, Y):
    row, col = X.shape
    pos = np.random.randint(0, row, (row,))
    return X[pos, :], Y[pos, :]


# In[15]:

# 随机森林算法---没有加入feature的随机选择
def random_forest(X, Y, T):
    nodeArr = []
    for i in range(T):
        Xtemp, Ytemp = bagging(X, Y)
        node = dTree(Xtemp, Ytemp)
        nodeArr.append(node)
    return nodeArr


# In[16]:

# 基于剪枝后的随机森林算法
def random_forest_pruned(X, Y, T):
    nodeArr = []
    for i in range(T):
        Xtemp, Ytemp = bagging(X, Y)
        node = dTree_one(Xtemp, Ytemp)
        nodeArr.append(node)
    return nodeArr


# In[17]:

# ----------------具体题目-------------------
# 加载数据函数
def loadData(filename):
    data = pd.read_csv(filename, sep='\s+', header=None)
    data = data.as_matrix()
    col, row = data.shape
    X = data[:, 0: row - 1]
    Y = data[:, row - 1:row]
    return X, Y


# In[18]:

# 导入数据
X, Y = loadData('hw3_train.dat')
Xtest, Ytest = loadData('hw3_test.dat')


# In[19]:

# Q13
# 定义一个搜索树有多少结点的函数---叶子结点不计入
def internal_node(node):
    if node == None:
        return 0
    if node.leftNode == None and node.rightNode == None:
        return 0
    l = 0;
    r = 0
    if node.leftNode != None:
        l = internal_node(node.leftNode)
    if node.rightNode != None:
        r = internal_node(node.rightNode)
    return 1 + l + r


node = dTree(X, Y)
print('完全生长的决策树内部结点数目：', internal_node(node))

# In[22]:

# Q14 and Q15
_, ein = err_fun(X, Y, node)
_, eout = err_fun(Xtest, Ytest, node)
print('Ein: ', ein, '\nEout: ', eout)

# In[27]:

# Q16,Q17,Q18
ein = 0;
eout = 0;
err = 0
for j in range(50):
    nodeArr = random_forest(X, Y, 300)
    l = len(nodeArr)
    yhat1 = np.zeros((Y.shape[0], l))
    yhat2 = np.zeros((Ytest.shape[0], l))
    for i in range(l):
        yhat1[:, i:i + 1], _ = err_fun(X, Y, nodeArr[i])
        yhat2[:, i:i + 1], _ = err_fun(Xtest, Ytest, nodeArr[i])
    errg = np.sum(yhat1 != Y, 0) / Y.shape[0]
    Yhat = np.sign(np.sum(yhat1, 1)).reshape(Y.shape)
    Ytesthat = np.sign(np.sum(yhat2, 1)).reshape(Ytest.shape)
    Yhat[Yhat == 0] = 1;
    Ytesthat[Ytesthat == 0] = 1
    ein += np.sum(Yhat != Y) / Y.shape[0]
    eout += np.sum(Ytesthat != Ytest) / Ytest.shape[0]
    err += np.sum(errg) / l
print('Ein(gt)的平均：', err / 50)
print('Ein(G): ', ein / 50)
print('Eout(G): ', eout / 50)

# In[28]:

# Q19, Q20
ein = 0;
eout = 0
for j in range(50):
    nodeArr = random_forest_pruned(X, Y, 300)
    l = len(nodeArr)
    yhat1 = np.zeros((Y.shape[0], l))
    yhat2 = np.zeros((Ytest.shape[0], l))
    for i in range(l):
        yhat1[:, i:i + 1], _ = err_fun(X, Y, nodeArr[i])
        yhat2[:, i:i + 1], _ = err_fun(Xtest, Ytest, nodeArr[i])
    Yhat = np.sign(np.sum(yhat1, 1)).reshape(Y.shape)
    Ytesthat = np.sign(np.sum(yhat2, 1)).reshape(Ytest.shape)
    Yhat[Yhat == 0] = 1;
    Ytesthat[Ytesthat == 0] = 1
    ein += np.sum(Yhat != Y) / Y.shape[0]
    eout += np.sum(Ytesthat != Ytest) / Ytest.shape[0]
print('Ein: ', ein / 50)
print('Eout: ', eout / 50)
