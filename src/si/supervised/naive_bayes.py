import numpy as np
from ..util import accuracy_score

__all__ = ['GaussianNB']


class GaussianNB:

	# https://medium.com/@rangavamsi5/na%C3%AFve-bayes-algorithm-implementation-from-scratch-in-python-7b2cc39268b9
	def __init__(self):

		"""
			Attributes:
				likelihoods: Likelihood of each feature per class
				class_priors: Prior probabilities of classes
				pred_priors: Prior probabilities of features
				features: All features of dataset
		"""
		super(GaussianNB, self).__init__()
		self.features = list
		self.likelihoods = {}
		self.class_priors = {}
		self.pred_priors = {}

		self.X_train = np.array
		self.y_train = np.array
		self.train_size = int
		self.num_feats = int

	def fit(self, X):

		self.features = list(X._xnames)
		self.X_train = X.X
		self.y_train = X.Y
		self.train_size = X.X.shape[0]
		self.num_feats = X.X.shape[1]
		self.is_fitted = True

		for feature in range(len(self.features)):
			self.likelihoods[self.features[feature]] = {}
			self.pred_priors[self.features[feature]] = {}

			for feat_val in np.unique(self.X_train[:, feature:feature + 1]):
				self.pred_priors[self.features[feature]].update({feat_val: 0})

				for outcome in np.unique(self.y_train):
					self.likelihoods[self.features[feature]].update({str(feat_val) + '_' + outcome: 0})
					self.class_priors.update({outcome: 0})

		self._calc_class_prior()
		self._calc_likelihoods()
		self._calc_predictor_prior()

	def _calc_class_prior(self):

		""" P(c) - Prior Class Probability """

		for outcome in np.unique(self.y_train):
			outcome_count = sum(self.y_train == outcome)
			self.class_priors[outcome] = outcome_count / self.train_size

	def _calc_likelihoods(self):

		""" P(x|c) - Likelihood """

		for feature in range(len(self.features)):

			for outcome in np.unique(self.y_train):
				outcome_count = sum(self.y_train == outcome)
				idx = self.y_train == str(outcome)
				feat_likelihood_prep = self.X_train[:, feature:feature + 1][idx]
				feat, count = np.unique(feat_likelihood_prep, return_counts=True)
				feat_likelihood = {}
				for i in range(len(feat)):
					feat_likelihood[feat[i]] = count[i]

				for feat_val, count in feat_likelihood.items():
					self.likelihoods[self.features[feature]][str(feat_val) + '_' + outcome] = count / outcome_count

	def _calc_predictor_prior(self):

		""" P(x) - Evidence """

		for feature in range(len(self.features)):
			feat, count = np.unique(self.X_train[:, feature:feature + 1], return_counts=True)
			feat_vals = {}

			for i in range(len(feat)):
				feat_vals[feat[i]] = count[i]

			for feat_val, count in feat_vals.items():
				self.pred_priors[self.features[feature]][feat_val] = count / self.train_size

	def predict(self, X):

		""" Calculates Posterior probability P(c|x) """

		assert self.is_fitted, 'Model must be fit before predict'

		results = []
		X = np.array(X)

		for query in X:
			probs_outcome = {}
			for outcome in np.unique(self.y_train):
				prior = self.class_priors[outcome]
				likelihood = 1
				evidence = 1

				for feat, feat_val in zip(self.features, query):
					likelihood *= self.likelihoods[feat][str(feat_val) + '_' + outcome]
					evidence *= self.pred_priors[feat][feat_val]

				posterior = (likelihood * prior) / (evidence)

				probs_outcome[outcome] = posterior

			result = max(probs_outcome, key=lambda x: probs_outcome[x])
			results.append(result)
			self.results = np.array(results)
		return self.results

	def cost(self, Y_test):
		return accuracy_score(Y_test, self.results)