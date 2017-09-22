from svmutil import *

y, x = [-1, -1, -1, 1, 1, 1, 1], [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
prob  = svm_problem(y, x, isKernel=True)
param = svm_parameter('-t 1 -g 1 -r 2 -d 2')
m = svm_train(prob, param)

print(m.get_SV())


# from cvxopt import matrix
# A = matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2,3))


# import numpy as np

# def kernel(x_n, x_m):
# 	return (2 + np.dot(np.transpose(x_n), x_m)) ** 2


# y = np.array([-1, -1, -1, 1, 1, 1, 1])
# x = np.array([
# 	[1, 0], 
# 	[0, 1], 
# 	[0, -1], 
# 	[-1, 0], 
# 	[0, 2], 
# 	[0, -2], 
# 	[-2, 0]])
# z = [[-3, -2], [3, 5], [3, -1], [5, -2], [9, -7], [9, 1], [9, 1]]

# len = y.shape[0]
# QMatrix = np.zeros((len, len))

# for n in range(len):
# 	for m in range(len):
# 		QMatrix[n, m] = y[n] * y[m] * kernel(x[n], x[m])

# print(QMatrix)
