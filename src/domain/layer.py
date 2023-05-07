import numpy as np

from enum import Enum

# ------------------------- ACTIVATION FUNCTION -------------------------

# Enum storing available activation functions
class Activation(Enum):
    SIGMOID = lambda input: 1/(1 + np.exp(-input))
    SOFTMAX = lambda input: np.exp(input) / np.sum(np.exp(input), keepdims=True)
    RELU = lambda input: np.multiply(input, input > 0)
    TANH = lambda input: np.tanh(input)
    LINEAR = lambda input: input

# Enum storing derivatives of available activation functions
class ActivationDer(Enum):
    SIGMOID = lambda input: Activation.SIGMOID(input) * (1 - Activation.SIGMOID(input))
    SOFTMAX = lambda input: Activation.SOFTMAX(input) * (1 - Activation.SOFTMAX(input))
    RELU = lambda input: input > 0
    TANH = lambda input: 1 - Activation.TANH(input)**2
    LINEAR = lambda _: 1

# Enum storing pairs of activation functions (i.e. activation function + its derivative)
# 
# Parameters:
# input - ndarray input signal
# der - boolean indicating wheteher the activation function or its derivative is to be computed
class LayerActivation(Enum):
    SIGMOID = lambda input, der: Activation.SIGMOID(input) if not der else ActivationDer.SIGMOID(input)
    SOFTMAX = lambda input, der: Activation.SOFTMAX(input) if not der else ActivationDer.SOFTMAX(input)
    RELU = lambda input, der: Activation.RELU(input) if not der else ActivationDer.RELU(input)
    TANH = lambda input, der: Activation.TANH(input) if not der else ActivationDer.TANH(input)
    LINEAR = lambda input, der: Activation.LINEAR(input) if not der else ActivationDer.LINEAR(input)

# ------------------------- NETWORK LAYER ---------------------------

class Layer:
    # Method initialize the layer
    #
    # Parameters:
    # neurons - number of neurons for a given layer
    # activation - type of the activation function
    def __init__(self, neurons: int, activation: LayerActivation) -> None:
        self.input_size: int
        self.output_size: int = neurons
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.activation = activation

    # Method constructs the layer
    #
    # Parameters:
    # input_size - size of the input signal that comes to the neurons in the layer
    def __construct__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.rand(self.output_size, self.input_size)
        self.biases = np.random.rand(self.output_size)

    # Method triggers activation of neurons in the layer
    #
    # Parameters:
    # inputs - input received by the layer
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs) + self.biases, der=False)
    
    # Method updates the weights in a given layer
    #
    # Parameters:
    # weight_diff - ndarray of differences in weights 
    # bias_diff   - ndarray of differences in weights of biases
    def __update_weights__(self, weight_diff: np.ndarray, bias_diff: np.ndarray):
        self.weights = self.weights - weight_diff.reshape(self.output_size, self.input_size)
        self.biases = self.biases - bias_diff

class InputLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, LayerActivation.LINEAR)
        self.input_size = input_size
        self.weights = np.ones((self.output_size, self.input_size))
        self.biases = np.zeros((self.output_size))
    
    def __construct__(self, input_size):
        return

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def __update_weights__(self, weight_diff: np.ndarray, bias_diff: np.ndarray):
        return