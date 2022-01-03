import itertools

# Y is reserved to idenfify dependent variables
import numpy as np
import pandas as pd

ALPHA = 'ABCDEFGHIJKLMNOPQRSTUVWXZ'

__all__ = ['label_gen', 'euclidian_distance', 'manhattan_distance', 'train_test_split', 'sigmoid', 'add_intersect']


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


def l1_distance(x, y):
    """Computes the manhatan distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)
    """
    import numpy as np
    dist = (np.absolute(x - y)).sum(axis=1)
    return dist


def l2_distance(x, y):
    """Computes the euclidean distance of a point (x) to a set of
    points y.
    x.shape=(n,) and y.shape=(m,n)
    """
    dist = ((x - y) ** 2).sum(axis=1)
    return dist


def euclidian_distance(x, y):
    dist = np.sqrt(((x -y) ** 2).sum(axis=1))
    return dist


def manhattan_distance(x, y):
    dist = np.absolute(x-y).sum(axis=1)
    return dist


def train_test_split(dataset, split=0.8):
    numtst = dataset.X.shape[0]
    arr = np.arange(numtst)
    m = int(split*numtst)
    np.random.shuffle(arr)
    from si.data import Dataset
    train = Dataset(dataset.X[arr[:m]], dataset.Y[arr[:m]], dataset._xnames, dataset._yname)
    test = Dataset(dataset.X[arr[m:]], dataset.Y[arr[m:]], dataset._xnames, dataset._yname)
    return train, test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def add_intersect(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def minibatch(X, batchsize=256, shuffle=True):
    N = X.shape[0]
    ix = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))

    if shuffle:
        np.random.shuffle(ix)

    def mb_generator():
        for i in range(n_batches):
            yield ix[i * batchsize: (i + 1) * batchsize]

    return mb_generator(),