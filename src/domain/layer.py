import numpy as np

from typing import Callable
from enum import Enum

# ------------------------- ACTIVATION FUNCTION -------------------------

# Enum storing available activation functions

class Activation(Enum):
    LINEAR: Callable[[np.ndarray], np.ndarray] = lambda input: input
    THRESHOLD: Callable[[np.ndarray],
                        np.ndarray] = lambda input: 2 * (input >= 0) - 1
    

class Weights(Enum):
    BIPOLAR: Callable[[np.ndarray], np.ndarray] = lambda input: input
    UNIPOLAR: Callable[[np.ndarray], np.ndarray] = lambda input: 2 * input - 1

# ------------------------- NETWORK LAYER -------------------------------


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

        self.weights = np.zeros(input_size)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """
        Triggers activation of neurons in the layer, computes layer output.

        Parameters:
        inputs - input received by the layer
        """
        raise NotImplementedError(
            f"{self} is missing implementation of __call__ method")

    def _initialize_weights(self, patterns: np.ndarray) -> None:
        """
        Initializes the layer weights (sets initial weights).

        Parameters:
        patterns - array containing patterns to put into hopfield memory
        """
        raise NotImplementedError(
            f"{self} is missing implementation of _initialize_weights method")


# ------------------------- LAYER INSTANCES ---------------------------
class InputLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _initialize_weights(self, patterns: np.ndarray) -> None:
        return


class HopfieldLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs))

    def _initialize_weights(self, patterns: np.ndarray) -> None:
        is_bipolar = (patterns == -1).any()
        transform_pattern = Weights.BIPOLAR if is_bipolar == True else Weights.UNIPOLAR
        pattern_matrix = transform_pattern(patterns)

        self.weights = (1 / self.weights.shape[0]) * pattern_matrix.T @ pattern_matrix
        np.fill_diagonal(self.weights, 0)


class OutputLayer(Layer):
    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _initialize_weights(self, patterns: np.ndarray) -> None:
        return


class InvisibleLayer(Layer):
    """
    Special layer used only by the GUI for drawing purposes.
    """

    def __init__(self, input_size: int) -> None:
        super().__init__(input_size, input_size, Activation.LINEAR)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def _initialize_weights(self, patterns: np.ndarray) -> None:
        return
