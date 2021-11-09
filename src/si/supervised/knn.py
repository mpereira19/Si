import numpy as np
from si.util import euclidian_distance, accuracy_score

__all__ = ['Knn']


class Knn:

	def __init__(self, number_neighbors, classification):
		super(Knn, self).__init__()
		self.k = number_neighbors
		self.classification = classification

	def fit(self, dataset):
		self.dataset = dataset
		self.is_fitted = True

	def get_neighbors(self, x):
		# calcular a distância do x a todos os outros pontos do nosso dataset
		dist = euclidian_distance(x, self.dataset.X)
		# ordenar os indices por ordem crescente de distância
		sorted_idxs = np.argsort(dist)
		return sorted_idxs[:self.k]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before prediction'
		# pegamos nos k mais próximos (mascara)
		# y_values = dataset.y[mascara]
		neighbors = self.get_neighbors(x)
		values = self.dataset[neighbors].tolist()
		if self.classification:
			prediction = max(set(values), key=values.count)
		else:
			prediction = sum(values)/len(values)
		return prediction

	def cost(self):
		y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
		return accuracy_score(self.dataset.Y, y_pred)
