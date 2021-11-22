from .modelo import Modelo
import numpy as np
from ..util import mse, sigmoid

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
		X = np.hstack((np.ones((X.shape[0], 1)), X))
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

	def cost(self):
		y_pred = np.dot(self.X, self.theta)
		return mse(self.Y, y_pred)/2


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
		X = np.hstack((np.ones((X.shape[0], 1)), X))
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

	def cost(self):
		y_pred = np.dot(self.X, self.theta)
		return mse(self.Y, y_pred)/2


class LogisticRegression:

	' Regressão Logística sem regularização'

	def __init__(self, gd=False, epochs=1000, lr=0.001):
		super(LogisticRegression, self).__init__()
		self.gd = gd
		self.theta = None
		self.epochs = epochs
		self.lr = lr

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = np.hstack((np.ones((X.shape[0], 1)), X))
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
			grad = np.dot(X.T, (h - y))
			self.theta -= self.lr * grad
			self.history[epoch] = [self.theta[:], self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predict'
		hs = np.hstack(([1], x))
		p = sigmoid(np.dot(self.theta, hs))
		if p >= 0.5: res = 1
		else: res = 0
		return res

	def cost(self):
		h = sigmoid(np.dot(self.X, self.theta))
		cost = (-self.Y * np.log(h) - (1 - self.Y) * np.log(1-h))
		res = np.sum(cost)/self.X.shape[0]
		return res


class LogisticRegressionReg:

	' Regressão Logística com regularização'

	def __init__(self, gd=False, epochs=1000, lr=0.001, lbd=1):
		super(LogisticRegressionReg, self).__init__()
		self.gd = gd
		self.theta = None
		self.epochs = epochs
		self.lr = lr
		self.lbd = lbd

	def fit(self, dataset):
		X, y = dataset.getXy()
		X = np.hstack((np.ones((X.shape[0], 1)), X))
		########
		# Só é necessário para fazer o score (cost) caso não queiram dar os dados
		self.X = X
		self.Y = y
		##########
		self.train(X, y)
		self.is_fitted = True

	def train(self, X, y):
		m, n = X.shape
		self.history = {}
		self.theta = np.zeros(n)
		for epoch in range(self.epochs):
			z = np.dot(X, self.theta)
			h = sigmoid(z)
			grad = np.dot(X.T, (h - y))
			reg = self.lbd / m * self.theta
			self.theta -= self.lr * (grad - reg)
			self.history[epoch] = [self.theta[:], self.cost()]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predict'
		hs = np.hstack(([1], x))
		p = sigmoid(np.dot(self.theta, hs))
		if p >= 0.5:
			res = 1
		else:
			res = 0
		return res

	def cost(self):
		h = sigmoid(np.dot(self.X, self.theta))
		cost = (-self.Y * np.log(h) - (1 - self.Y) * np.log(1 - h))
		res = np.sum(cost) / self.X.shape[0]
		return res