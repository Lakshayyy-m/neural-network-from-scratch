import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from loss import Loss_CategoricalCrossentropy
from activation import Activation_Softmax, Activation_ReLU
from utils import BaseLayer, calculate_accuracy

nnfs.init()

np.random.seed(0)


class Layer_Dense(BaseLayer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__()
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)

accuracy = calculate_accuracy(activation2.output, y)

print("Loss: ", loss)
print("Accuracy: ", accuracy)
