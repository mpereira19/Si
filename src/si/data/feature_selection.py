import numpy as np
from scipy import stats
from copy import copy
import warnings
from si.data import Dataset

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
		X, x_names = copy(dataset.X), copy(dataset._xnames)
		cond = self._var > self.threshold
		X_trans = X[:, cond]
		idxs = [i for i in range(dataset.getNumFeatures()) if cond[i]]
		xnames = [x_names[i] for i in idxs]
		if inline:
			dataset.X = X_trans
			dataset._xnames = xnames
			return dataset
		else:
			from .dataset import Dataset
			return Dataset(X_trans, copy(dataset.Y), xnames, copy(dataset._yname))

	def fit_transform(self, dataset, inline=False):
		self.fit(dataset)
		return self.transform(dataset, inline)

class selectKbest:

	def __init__(self, k, scoring_function):
		self.scor_func = scoring_function
		self.k = k

	def fit(self, dataset):
		self.f, self.pvalue = self.scor_func(dataset)

	def transform(self, dataset, inline=False):
		x, x_names = copy(dataset.X), copy(dataset._xnames)
		indexes = sorted(self.f.argsort()[len(self.f)-self.k:])
		new_data = x[:, indexes]
		xnames = [x_names[i] for i in indexes]
		if inline:
			dataset.X = new_data
			dataset._xnames = xnames
			return dataset
		else:
			from .dataset import Dataset
			return Dataset(new_data, copy(dataset.Y), xnames, copy(dataset._yname))

	def fit_transform(self, dataset, inline=False):
		self.fit(dataset)
		return self.transform(dataset, inline=inline)


def f_classification(dataset):
	from scipy.stats import f_oneway
	x, y = dataset.getXy()
	arguments = [x[cat == y, :] for cat in np.unique(y)]
	f_s, p = f_oneway(*arguments)
	return f_s, p


def f_regression(dataset):
	# https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
	# https://github.com/scikit-learn/scikit-learn/blob/844b4be24/sklearn/feature_selection/_univariate_selection.py#L294
	from scipy.stats import f

	x, y = dataset.getXy()
	corr = np.corrcoef(x, rowvar=False)

	# Apenas queremos a correlação de cada variável com o y daí utilizarmos a última linha da matriz sem o último
	# valor da linha.

	corr = corr[-1:, :len(corr)].reshape(x.shape[1])
	sq_corr = corr ** 2
	degree_freedom = y.size-2
	f_r = sq_corr/(1-sq_corr)*degree_freedom
	p = f.sf(f_r, 1, degree_freedom)
	return f_r, p


