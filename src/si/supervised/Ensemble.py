import numpy as np
from .modelo import Modelo

__all__ = ['Ensemble', 'majority', 'average']


def majority(values):
	return max(set(values), key=values.count)


def average(values):
	return sum(values)/len(values)


class Ensemble(Modelo):

	def __init__(self, models :list, fvote, score):

		"""fvote: O valor com maior frequência"""

		super().__init__()
		self.models = models
		self.fvote = fvote
		self.score = score

	def fit(self, dataset):
		self.dataset = dataset
		for model in self.models: model.fit(dataset)
		self.is_fitted = True

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before predicting'
		preds = [model.predict(x) for model in self.models]
		vote = self.fvote(preds)
		return vote

	def cost(self, X=None, y=None):
		X = X if X is not None else self.dataset.X
		y = y if y is not None else self.dataset.Y
		y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=X.T)
		return self.score(y, y_pred)


class ConfusionMatrix:

	def calc(self, true_y, pred_y):
		...
		self.conf = ...

	def toDataFrame(self):
		pass

	def __call__(self, true_y, pred_y):
		self.calc(true_y, pred_y)
