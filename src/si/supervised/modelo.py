from abc import ABC, abstractmethod

__all__ = ['Modelo']


class Modelo(ABC):

	def __init__(self):
		'''
		Abstract class defining an interface for supervised learning models.
		'''
		self.is_fitted = False

	@abstractmethod  # torna a função abstrata. Sempre que criarem uma suprclass no modelo ele verifica se estes métodos estão implementados.
	def fit(self, dataset):
		raise NotImplementedError

	@abstractmethod
	def predict(self, x):
		raise NotImplementedError

	@abstractmethod
	def cost(self):
		raise NotImplementedError
