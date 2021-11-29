import numpy as np
from si.util import euclidian_distance, accuracy_score, manhattan_distance

__all__ = ['Knn']


class Knn:

	def __init__(self, number_neighbors :int, classification=True, func=euclidian_distance):
		super(Knn, self).__init__()
		self.k = number_neighbors
		self.classification = classification
		if func == euclidian_distance or func == manhattan_distance:
			self.func = func
		else: raise Exception('Score functions: euclidean_distance, manhattan_distance')

	def fit(self, dataset):
		self.dataset = dataset
		self.is_fitted = True
		return self.dataset

	def get_neighbors(self, x):
		# calcular a distância do x a todos os outros pontos do nosso dataset
		dist = self.func(x, self.dataset.X)
		# ordenar os indices por ordem crescente de distância
		sorted_idxs = np.argsort(dist)
		return sorted_idxs[:self.k]

	def predict(self, x):
		assert self.is_fitted, 'Model must be fit before prediction'
		# pegamos nos k mais próximos (mascara)
		# y_values = dataset.y[mascara]
		neighbors = self.get_neighbors(x)
		values = self.dataset.Y[neighbors].tolist()
		if self.classification:
			prediction = max(set(values), key=values.count)
		else:
			prediction = sum(values)/len(values)
		return prediction

	def cost(self):
		y_pred = np.ma.apply_along_axis(self.predict, axis=0, arr=self.dataset.X.T)
		return accuracy_score(self.dataset.Y, y_pred)
