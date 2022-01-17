import numpy as np
from ..supervised import Modelo
from abc import ABC, abstractmethod
from ..util import mse, mse_prime, im2col, pad2D, col2im

__all__ = ['Layer', 'Dense', 'Activation', 'NN', 'Flatten', 'Conv2D', 'MaxPooling2D', 'Pooling2D']


class Layer(ABC):

    def __init__(self):
        self.inputData = None
        self.outputData = None

    @abstractmethod
    def forward(self, inputData):
        raise NotImplementedError
        # return output

    @abstractmethod
    def backward(self, error, lr):  # lr = LearningRate
        raise NotImplementedError
        # return input_error


class Dense(Layer):

    def __init__(self, inputSize, outputSize):
        super(Layer, self).__init__()
        self.weights = np.random.rand(inputSize, outputSize)  # Matriz de pesos
        self.bias = np.random.rand(1, outputSize)

    def setWeights(self, weights, bias):
        if weights.shape != self.weights.shape:
            raise ValueError(f'Shapes mismatch {weights.shape} and {self.weights.shape}')
        if bias.shape != self.bias.shape:
            raise ValueError(f'Shapes mismatch {bias.shape} and {self.bias.shape}')
        self.weights = weights
        self.bias = bias

    def forward(self, inputData):
        self.input = inputData
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_error, lr):
        # compute the weight error
        weights_error = np.dot(self.input.T, output_error)
        # compute the bias error
        bias_error = np.sum(output_error, axis=0)
        # compute dE/dX to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)

        self.weights -= lr * weights_error
        self.bias -= lr * bias_error
        return input_error


class Activation(Layer):

    def __init__(self, activFunc):
        super(Layer, self).__init__()
        self.func = activFunc  # Fucao de ativacao

    def forward(self, inputData):
        self.input = inputData
        self.output = self.func(self.input)
        return self.output

    def backward(self, output_error, lr):
        return np.multiply(self.func.prime(self.input), output_error)


class Flatten(Layer):

    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, inputData):
        self.input_shape = inputData.shape
        output = inputData.reshape(inputData.shape[0], -1)
        return output

    def backward(self, error, lr):
        return error.reshape(self.input_shape)


class Conv2D(Layer):

    def __init__(self, input_shape, kernel_shape, layer_depth, stride=1, padding=0):
        super(Layer, self).__init__()
        self.input_shape = input_shape
        self.in_ch = input_shape[2]
        self.out_ch = layer_depth
        self.stride = stride
        self.padding = padding

        # Weights
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1], self.in_ch, self.out_ch) - 0.5

        # Bias
        self.bias = np.zeros((self.out_ch, 1))

    def forward(self, inputData):
        s = self.stride
        self.X_shape = inputData.shape
        _, p = pad2D(inputData, self.padding, self.weights.shape[:2], s)

        pr1, pr2, pc1, pc2 = p
        fr, fc, in_ch, out_ch = self.weights.shape
        n_ex, in_rows, in_cols, in_ch = inputData.shape

        # compute the dimensions of convolution output
        out_rows = int((in_rows + pr1 + pr2 - fr) / s + 1)
        out_cols = int((in_cols + pc1 + pc2 - fc) / s + 1)

        # convert X and W into the appropriate 2D matrices and take their product
        self.X_col, _ = im2col(inputData, self.weights.shape, p, s)
        W_col = self.weights.transpose(3, 2, 0, 1).reshape(out_ch, -1)

        output_data = (W_col @ self.X_col + self.bias).reshape(out_ch, out_rows, out_cols, n_ex).transpose(3, 1, 2, 0)

        return output_data

    def backward(self, error, lr):
        fr, fc, in_ch, out_ch = self.weights.shape
        p = self.padding

        db = np.sum(error, axis=(0, 1, 2))
        db = db.reshape(out_ch, )

        dout_reshaped = error.transpose(1, 2, 3, 0).reshape(out_ch, -1)
        dW = dout_reshaped @ self.X_col.T
        dW = dW.reshape(self.weights.shape)

        W_reshape = self.weights.reshape(out_ch, -1)
        dX_col = W_reshape.T @ dout_reshaped
        input_error = col2im(dX_col, self.X_shape, self.weights.shape, (p, p, p, p), self.stride)

        self.weights -= lr * dW
        self.bias -= lr * db

        return input_error


class NN(Modelo):

    def __init__(self, epochs=1000, lr=0.01, verbose=False):
        super(Modelo, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose

        self.layers = []
        self.loss = mse
        self.loss_prime = mse_prime

    def fit(self, dataset):
        self.dataset = dataset
        self.history = dict()
        for epoch in range(self.epochs):
            output = dataset.X

            # forward propagation
            for layer in self.layers:
                output = layer.forward(output)

            # backward propagation
            error = self.loss_prime(dataset.y, output)
            for layer in reversed(self.layers):
                error = layer.backward(error, self.lr)

            # calculate average error all samples
            err = self.loss(dataset.y, output)
            self.history[epoch] = err

            if self.verbose:
                print(f'epoch {epoch + 1}/{self.epochs} error = {err}')

        print(f'error = {err}')

        self.is_fitted = True

    def useLoss(self, func, func2):
        self.loss, self.loss_prime = func, func2

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def cost(self, X=None, y=None):
        X = X
        y = y
        output = self.predict(X)
        return self.loss(y, output)


class Pooling2D(Layer):

    def __init__(self, size=2, stride=2):
        super().__init__()
        self.X_shape = None
        self.size = size
        self.stride = stride

    def pool(self, X_col):
        raise NotImplementedError

    def dpool(self, dX_col, dout_col, pool_cache):
        raise NotImplementedError

    def forward(self, input_data):
        self.X_shape = input_data.shape
        n, h, w, d = input_data.shape

        h_out = int((h - self.size) // self.stride + 1)
        w_out = int((w - self.size) // self.stride + 1)

        if not type(w_out) is int or not type(h_out) is int:
            raise Exception('Invalid Output Dim')

        X_reshape = input_data.reshape(n * d, h, w, 1)
        self.X_col, _ = im2col(X_reshape, (self.size, self.size, 1, 1), pad=0, stride=self.stride)

        out, self.max_idx = self.pool(self.X_col)

        out = out.reshape((h_out, w_out, n, d))
        out = out.transpose(2, 0, 1, 3)

        return out

    def backward(self, output_error, lr):
        n, h, w, d = self.X_shape
        dX_col = np.zeros_like(self.X_col)
        dout_col = output_error.transpose(1, 2, 3, 0).ravel()

        dX_col = self.dpool(dX_col, dout_col, self.max_idx)

        dX = col2im(dX_col, (n * d, h, w, 1), (self.size, self.size, 1, 1), pad=(0, 0, 0, 0), stride=self.stride)
        dX = dX.reshape(self.X_shape)

        return dX


class MaxPooling2D(Pooling2D):

    def pool(self, X_col):
        out = np.amax(X_col, axis=0)
        max_indx = np.argmax(X_col, axis=0)
        return out, max_indx

    def dpool(self, dX_col, dout_col, pool_cache):
        for x, indx in enumerate(pool_cache):
            dX_col[indx, x] = 1
        return dX_col * dout_col
