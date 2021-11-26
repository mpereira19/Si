from .modelo import Modelo
import numpy as np
from ..util import mse, sigmoid, add_intersect
from ..supervised import modelo

__all__ = ['LinearRegression', 'LinearRegressionReg', 'LogisticRegression', 'LogisticRegressionReg']


class LinearRegression:

	"Regressão linear sem regularização"

	def __init__(self, gd=False, epochs=1000, lr=0.001):
		super(LinearRegression, self).__init__()
		self.gd = gd
		self.theta = None
		self.epochs = epochs
		self.lr = lr

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = add_intersect(X)
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		# closed form or GD
		self.train_gd(X, y) if self.gd else self.train_closed(X, y)
		self.is_fitted = True

	def train_closed(self, X, y):
		self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

	def train_gd(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		for epoch in range(self.epochs):
			grad = 1/m * (X.dot(self.theta) - y).dot(X)
			self.theta -= self.lr * grad
			self.history[epoch] = [self.theta[:], self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predict'
		_x = np.hstack(([1], x))
		return np.dot(self.theta, _x)

	def cost(self, X=None, y=None, theta=None):
		X = add_intersect(X) if X is not None else self.X
		y = y if y is not None else self.Y
		theta = theta if theta is not None else self.theta
		y_pred = np.dot(X, theta)
		return mse(y, y_pred)/2


class LinearRegressionReg:

	"Regressão linear com regularização"

	def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
		super(LinearRegressionReg, self).__init__()
		self.gd = gd
		self.theta = None
		self.epochs = epochs
		self.lr = lr
		self.lbd = lbd

	def train_closed(self, X, y):
		m, n = X.shape
		identity = np.eye(n)
		identity[0, 0] = 0
		self.theta = np.linalg.inv(X.T.dot(X) + self.lbd * identity).dot(X.T).dot(y)

	def train_gd(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		lbds = np.full(m, self.lbd)
		lbds[0] = 0
		for epoch in range(self.epochs):
			grad = (X.dot(self.theta) - y).dot(X)
			self.theta -= (self.lr/m) * (lbds + grad)
			self.history[epoch] = [self.theta[:], self.cost()]

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = add_intersect(X)
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		# closed form or GD
		self.train_gd(X, y) if self.gd else self.train_closed(X, y)
		self.is_fitted = True

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predict'
		_x = np.hstack(([1], x))
		return np.dot(self.theta, _x)

	def cost(self, X=None, y=None, theta=None):
		X = add_intersect(X) if X is not None else self.X
		y = y if y is not None else self.Y
		theta = theta if theta is not None else self.theta
		y_pred = np.dot(X, theta)
		return mse(y, y_pred)/2


class LogisticRegression:

	' Regressão Logística sem regularização'

	def __init__(self, epochs=1000, lr=0.1):
		super(LogisticRegression, self).__init__()
		self.theta = None
		self.epochs = epochs
		self.lr = lr

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = add_intersect(X)
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		# closed form or GD
		self.train(X, y)
		self.is_fitted = True

	def train(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		for epoch in range(self.epochs):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			grad = np.dot(X.T, (h - y)) / y.size
			self.theta -= self.lr * grad
			self.history[epoch] = [self.theta.copy(), self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predicting'
		hs = np.hstack(([1], x))
		p = sigmoid(np.dot(self.theta, hs))
		if p >= 0.5: res = 1
		else: res = 0
		return res

	def cost(self, X=None, y=None, theta=None):
		X = add_intersect(X) if X is not None else self.X
		y = y if y is not None else self.Y
		theta = theta if theta is not None else self.theta
		m, n = X.shape
		h = sigmoid(np.dot(X, theta))
		cost = (-y * np.log(h) - (1 - y) * np.log(1-h))
		res = np.sum(cost) / m
		return res


class LogisticRegressionReg:

	' Regressão Logística com regularização'

	def __init__(self, epochs=1000, lr=0.1, lbd=1):
		super(LogisticRegressionReg, self).__init__()
		self.theta = None
		self.epochs = epochs
		self.lr = lr
		self.lbd = lbd  # lbd = lambda

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = add_intersect(X)
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		# closed form or GD
		self.train(X, y)
		self.is_fitted = True

	def train(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		for epoch in range(self.epochs):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			grad = np.dot(X.T, (h - y)) / y.size
			reg = (self.lbd / m) * self.theta[1:]  ###### parentesis
			grad[1:] = grad[1:] + reg
			self.theta -= self.lr * grad
			self.history[epoch] = [self.theta[:], self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predicting'
		hs = np.hstack(([1], x))
		p = sigmoid(np.dot(self.theta, hs))
		if p >= 0.5: res = 1
		else: res = 0
		return res

	def cost(self, X=None, y=None, theta=None):
		X = add_intersect(X) if X is not None else self.X
		y = y if y is not None else self.Y
		theta = theta if theta is not None else self.theta
		m = X.shape[0]
		h = sigmoid(np.dot(X, theta))
		cost = (-y * np.log(h) - (1 - y) * np.log(1 - h))
		reg = np.dot(theta[1:], theta[1:]) * self.lbd / (2 * m)
		res = np.sum(cost) / m
		return res + reg
