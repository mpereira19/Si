import numpy as np
from si.util.scale import StandardScaler as sc

__all__ = ['PCA']

class PCA:
	def __init__(self, n_comp, PCA_type='svd'):
		self.type = PCA_type
		if n_comp > 0 and isinstance(n_comp, int): self.n_comp = n_comp
		else: Warning('Number of components must an non negative integer.')

	def fit(self):
		pass

	def transform(self, dataset):
		scaled_feature = sc().fit_transform(dataset).X.T  # Normalização com a Classe Standard Scaler
		if self.type.lower() == 'svd': self.u, self.s, self.vh = np.linalg.svd(scaled_feature)
		else:
			self.cov_matrix = np.cov(scaled_feature)
			self.s, self.u = np.linalg.eig(self.cov_matrix)

		self.idxs = np.argsort(self.s)[::-1]  # gera um array com os indexes das colunas ordenadas por importância de compontes
		self.eigen_val, self.eigen_vect = self.s[self.idxs], self.u[:, self.idxs]  # colunas dos valores e dos vetores são reordenadas pelos indexes das colunas
		self.sub_set_vect = self.eigen_vect[:, :self.n_comp]  # gera um conjunto a partir dos vetores e values ordenados
		return scaled_feature.T.dot(self.sub_set_vect)

	def explained_variances(self):
		sum_ = np.sum(self.eigen_val)
		percentage = [i / sum_ * 100 for i in self.eigen_val]
		return np.array(percentage)

	def fit_transform(self, dataset):
		trans = self.transform(dataset)
		exp = self.explained_variances()
		return trans, exp
