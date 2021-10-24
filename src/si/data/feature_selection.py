import numpy as np
from scipy import stats
from copy import copy
import warnings

class VarianceThreshold:

	def __init__(self, threshold=0):
		if threshold < 0:
			warnings.warn('The threshold must be a non-negative value')
			threshold = 0
		self.threshold = threshold

	def fit(self, dataset):
		x = dataset.X
		self._var = np.var(x, axis=0)

	def transform(self, dataset, inline=False):
		X = dataset.X
		cond = self._var > self.threshold
		idxs = [i for i in range(len(cond)) if cond[i]]
		X_trans = X[:, idxs]
		xnames = [dataset._names[i] for i in idxs]
		if inline:
			dataset.X = X_trans
			dataset._xnames = xnames
			return dataset
		else:
			from .dataset import Dataset
			return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset.yname))

	def fit_transform(self, dataset, inline=False):
		self.fit(dataset)
		return self.transform(dataset, inline=inline)


class selectKbest:

	def __init__(self, scoring_function, k):
		self.scor_func = scoring_function
		self.k = k

	def fit(self, dataset):
		self.f, self.pvalue = self.scor_func(dataset)

	def transform(self, dataset, inline=False):
		indexes = self.f.argsort()[self.f.shape[1]-self.k:]
		indexes = indexes.sort()
		new_data = self.f[:, indexes]
		xnames = [dataset._names[i] for i in indexes]
		if inline:
			dataset.X = new_data
			dataset._xnames = xnames
			return dataset
		else:
			from .dataset import Dataset
			return Dataset(new_data, copy(dataset.Y), xnames, copy(dataset.yname))


def f_classification(dataset):
	from scipy.stats import f_oneway
	x, y = dataset.X, dataset.Y
	arguments = [x[cat == y, :] for cat in np.unique(y)]
	return f_oneway(arguments)


def f_regression(dataset):
	# https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
	# https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/feature_selection/_univariate_selection.py#L294
	from scipy.stats import f

	x, y = dataset.X, dataset.Y
	corr = np.corrcoef(x, rowvar=False)

	# Apenas queremos a correlação de cada variável com o y daí utilizarmos a última linha da matriz sem o último
	# valor da linha.

	corr = corr[-1:, :len(corr)-1]
	sq_corr = corr**2
	degree_freedom = y.size-2
	f_r = sq_corr/(1-sq_corr)*degree_freedom
	p = f.sf(f_r, 1, degree_freedom)
	return f_r, p


