import numpy as np
from si.util.util import l2_distance

class Kmeans:

	def __init__(self, k: int, iterations=100):
		self.k = k
		self.max_iter = iterations
		self.centroids = None
		self.distance = l2_distance

	def fit(self, dataset):
		x = dataset.X
		self._min = np.min(x, axis=0)
		self._max = np.max(x, axis=0)

	def generate_random_centroid(self, dataset):
		x = dataset.X
		self.centroids = np.array([np.random.uniform(low=self._min[i], high=self._max[i], size=(self.k,)) for i in range(x.shape[1])]).T

	def closest_centroid(self, x):
		dist = self.distance(x, self.centroids)
		closest_centroid = np.argmin(dist, axis=0)
		return closest_centroid

	def transform(self, dataset):
		self.generate_random_centroid(dataset)
		x = dataset.X
		changed = True
		count = 0
		old_idxs = np.zeros(x.shape[0])
		while changed and count < self.max_iter:
			idxs = np.apply_along_axis(self.closest_centroid, axis=0, arr=x.T)
			self.centroids = np.array([np.mean(x[idxs == i]) for i in range(self.k)])
			changed = np.all(old_idxs == self.idxs)
			old_idxs = idxs
			count += 1
		return self.centroids, idxs

	def fit_transform(self, dataset):
		self.fit(dataset)
		return self.transform(dataset)
