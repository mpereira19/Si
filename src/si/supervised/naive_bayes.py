import numpy as np
from si.util import euclidian_distance, accuracy_score

__all__ = ['GaussianNB']


class GaussianNB:

	def __init__(self, prior=None, var_smoothing=1e-9):
		super(GaussianNB, self).__init__()
		self.prior = prior
		self.var_smoothing = var_smoothing

	def fit(self, X, Y):
		self.n_examples, self.n_features = X.shape
		self.n_classes = len(np.unique(Y))
		self.class_id = np.unique(Y)
		self.is_fitted = True

		self.mean_classes = {str(cls): np.mean(X[Y == cls]) for cls in range(self.class_id)}
		self.var_classes = {str(cls): np.var(X[Y == cls]) for cls in range(self.class_id)}
		if self.prior is None:
			self.prior = {str(cls): X[Y == cls]/self.n_examples for cls in range(self.class_id)}

	def density(self, X, mean, sigma):
		constant = -self.n_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.var_smoothing))
		probability = 0.5 * np.sum(np.power(X - mean, 2) / (sigma + self.var_smoothing), 1)
		return constant - probability

	def predict(self, x):
		probability = np.zeros(self.n_examples, self.n_features)
		for cls in range(self.class_id):
			probability_c = self.density(x, self.mean_classes[str(cls)], self.var_classes[str(cls)])
			probability[:, cls] = probability_c + np.log(self.prior[str(cls)])
		return np.argmax(probability, 1)

	def cost(self):
		pass