import numpy as np
from ..supervised import Modelo
from scipy import signal
from abc import ABC, abstractmethod
from ..util import mse, mse_prime

__all__ = ['Dense', 'Activation', 'NN']


class Layer(ABC):

	def __init__(self):
		self.input = None
		self.output = None

	@abstractmethod
	def forward(self, input):
		raise NotImplementedError

	@abstractmethod
	def backward(self, output_error, learning_rate):
		raise NotImplementedError


class Dense(Layer):

	def __init__(self, input_size, output_size):
		self.weights = np.random.rand(input_size, output_size) - 0.5
		self.bias = np.zeros((1, output_size))

	def setWeights(self, weights, bias):
		if weights.shape != self.weights.shape:
			raise ValueError(f'Shapes mismatch {weights.shape} and {self.weights.shape}')
		if bias.shape != self.bias.shape:
			raise ValueError(f'Shapes mismatch {bias.shape} and {self.bias.shape}')
		self.weights = weights
		self.bias = bias

	def forward(self, input_data):
		self.input = input_data
		self.output = np.dot(self.input, self.weights) + self.bias
		return self.output

	def backward(self, output_error, learning_rate):
		# compute the weights error dE/dW = X.T * dE/dY
		weights_error = np.dot(self.input.T, output_error)
		# compute the bias error dE/dB = dE/dY
		bias_error = np.sum(output_error, axis=0)
		# error dE/dX to pass on to the previous layer
		input_error = np.dot(output_error, self.weights.T)
		# update parameters
		self.weights -= learning_rate * weights_error
		self.bias -= learning_rate * bias_error
		return input_error


class Activation(Layer):

	def __init__(self, function):
		self.function = function

	def forward(self, input_data):
		self.input = input_data
		self.output = self.function(self.input)
		return self.output

	def backward(self, output_error, learning_rate):
		# learning rate is not used because is no "learnable" parameters,
		# Only passed the error do the previous layer
		return np.multiply(self.function.prime(self.input), output_error)


class NN(Modelo):
	def __init__(self, epochs=1000, lr=0.001, verbose=True):
		self.epochs = epochs
		self.lr = lr
		self.verbose = verbose

		self.layers = []
		self.loss = mse
		self.loss_prime = mse_prime

	def fit(self, dataset):
		X, y = dataset.getXy()
		self.dataset = dataset
		self.history = dict()
		for epoch in range(self.epochs):
			output = X

			# forward propagation
			for layer in self.layers:
				output = layer.forward(output)

			# backward propagation
			error = self. loss_prime(y, output)
			for layer in reversed(self.layers):
				error = layer.backward(error, self.lr)

			# calculate average error
			err = self.loss(y, output)
			self.history[epoch] = err
			if self.verbose:
				print(f'epoch {epoch+1}/{self.epochs} error={err}')
		if not self.verbose:
			print(f'error={err}')
		self.is_fitted = True

	def add(self, layer):
		self.layers.append(layer)

	def predict(self, x):
		self.is_fitted = True
		output = x
		for layer in self.layers:
			output = layer.forward(output)
		return output

	def cost(self, X=None, y=None):
		assert self.is_fitted, 'Model must be fit before predict'
		X = X if X is not None else self.dataset.X
		y = y if y is not None else self.dataset.Y
		output = self.predict(X)
		return self.loss(y, output)
