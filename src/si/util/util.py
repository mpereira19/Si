import itertools

# Y is reserved to idenfify dependent variables
import numpy as np
import pandas as pd

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'euclidian_distance', 'manhattan_distance', 'train_test_split']


def label_gen(n):
    """ Generates a list of n distinct labels similar to Excel"""
    def _iter_all_strings():
        size = 1
        while True:
            for s in itertools.product(ALPHA, repeat=size):
                yield "".join(s)
            size += 1

    generator = _iter_all_strings()

    def gen():
        for s in generator:
            return s

    return [gen() for _ in range(n)]


def euclidian_distance(x, y):
    dist = np.sqrt(((x -y) ** 2).sum(axis=1))
    return dist


def manhattan_distance(x, y):
    dist = np.absolute(x-y).sum(axis=1)
    return dist


def train_test_split(dataset, split=0.8):
    numtst = dataset.shape[0]
    arr = np.arange(numtst)
    m = int(split*numtst)
    np.random.shuffle(arr)
    from ..data import Dataset
    train = Dataset(dataset.X[arr[:m]], dataset.Y[arr[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[arr[m:]], dataset.Y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test