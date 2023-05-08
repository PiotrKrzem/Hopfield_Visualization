import numpy as np

from typing import Callable
from enum import Enum

# ------------------------- ACTIVATION FUNCTION -------------------------

# Enum storing available activation functions
class Activation(Enum):
    LINEAR: Callable[[np.ndarray], np.ndarray] = lambda input: input
    THRESHOLD: Callable[[np.ndarray], np.ndarray] = lambda input: 2 * (input >= 0) - 1

# ------------------------- NETWORK LAYER ---------------------------
class Layer:
    def __init__(self, input_size: int, output_size: int, activation: Activation) -> None:
        """
        Initializes the layer.
        
        Parameters:
        input_size - number of input values
        input_size - number of neurons for a given layer, amount of output values
        activation - type of the activation function
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights: np.ndarray
        self._construct()

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Triggers activation of neurons in the layer, computes layer output.
        
        Parameters:
        inputs - input received by the layer
        """
        raise NotImplementedError(f"{self} is missing implementation of __call__ method")

    def _construct(self) -> None:
        """
        Constructs the layer (sets weights).
        """
        raise NotImplementedError(f"{self} is missing implementation of _construct method")

    def _update_weights(self, weights_diff: np.ndarray) -> None:
        """
        Method updates the weights in a given layer
        
        Parameters:
        weights_diff - ndarray of differences in weights
        """
        raise NotImplementedError(f"{self} is missing implementation of _update_weights method")

# ------------------------- LAYER INSTANCES ---------------------------
class InputLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _construct(self):
        return

    def _update_weights(self, weights_diff: np.ndarray):
        return

class HopfieldLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.THRESHOLD)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        output = self.activation(np.dot(self.weights, inputs))
        return output

    def _construct(self):
        self.weights = np.random.randn(self.output_size, self.input_size)

    def _update_weights(self, weights_diff: np.ndarray):
        # TODO
        self.weights = self.weights - weights_diff.reshape(self.output_size, self.input_size)

class OutputLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _construct(self):
        return

    def _update_weights(self, weights_diff: np.ndarray):
        return

class InvisibleLayer(Layer):
    """
    Special layer used only by the GUI for drawing purposes.
    """
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)
    
    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _construct(self):
        return

    def _update_weights(self, weights_diff: np.ndarray):
        return