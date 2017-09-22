import numpy as np


def gini(mu_plus):

	mu_minus = 1 - mu_plus

	return 1 - mu_plus ** 2 - mu_minus ** 2

def a(mu_plus):
	mu_minus = 1 - mu_plus

	return min([mu_plus, mu_minus])

def b(mu_plus):
	mu_minus = 1 - mu_plus
	return mu_plus * (1 - (mu_plus - mu_minus)) ** 2 + mu_minus * (-1 - (mu_plus - mu_minus)) ** 2

def c(mu_plus):
	mu_minus = 1 - mu_plus
	return -1 * mu_plus * np.log(mu_plus) + -1 * mu_minus * np.log(mu_minus)

def d(mu_plus):
	mu_minus = 1 - mu_plus
	return 1 - np.abs(mu_plus - mu_minus)


fun_arr = [gini, a, b, c, d]
mu_plus_arr = [0.1, 0.3]

for fun in fun_arr:
	print("--------------------------")
	impurity_arr = []
	for mu_plus in mu_plus_arr:
		impurity = fun(mu_plus)
		if len(impurity_arr) > 0:
			print(impurity / impurity_arr[-1])
		impurity_arr.append(impurity)
	print(impurity_arr)