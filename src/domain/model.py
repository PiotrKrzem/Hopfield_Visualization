import numpy as np
from typing import List, Tuple
from time import time
import matplotlib.pyplot as plt

from src.domain.layer import InputLayer, HopfieldLayer, InvisibleLayer, OutputLayer
from src.argparser import Arguments

from src.path_parser import get_file_path


# -------------------------- HOPFIELD MODEL ------------------------------
class Model:
    def __init__(self, input_size: int, name = "model") -> None:
        """
        Initializes the hopfield model
        
        Parameters:
        input_size - size of the input data
        name (optional) - name of the model
        """
        self.name = name
        self.input_size = input_size
        self.input_layer = InputLayer(input_size)
        self.hopfield_layer = HopfieldLayer(input_size)
        self.output_layer = OutputLayer(input_size)

        self.layers = [
            self.input_layer,
            InvisibleLayer(input_size),
            self.hopfield_layer,
            InvisibleLayer(input_size),
            self.output_layer
        ]

    def __call__(self, inputs: np.ndarray):
        """
        Compute outputs for given samples.

        Parameters:
        inputs: 2D array of samples of shape [count, sample]
        """
        if len(inputs.shape) != 2:
            raise Exception("inputs shape should be 2D and follow pattern [inputs_count, input_vector]")

        outputs = []

        # iterating over all samples
        for idx in range(len(inputs)):
            inputs_sample = inputs[idx]         

            # getting predicted output based on forward propagation
            outputs_sample = self._forward(inputs_sample)
            outputs.append(outputs_sample)

        return outputs

    def _forward(self, input: np.ndarray) -> List[np.ndarray]:
        """
        Method performs forward propagation
        
        Parameters:
        input - input data sent to the model (2D shape [count, sample])
        """
        outputs: List[np.ndarray] = []

        first_output = self.hopfield_layer(input)
        second_output = self.hopfield_layer(first_output)

        if np.array_equal(first_output, second_output):
            outputs = [first_output]
        else:
            outputs = [first_output, second_output]
        return outputs

    def train(self, patterns: np.ndarray) -> None:
        """
        Trains the model to remember given patterns.
        
        Parameters:
        patterns: array containing patterns to put into hopfield memory
        """
        self.hopfield_layer._initialize_weights(patterns)
    
    def test(self, inputs: np.ndarray = None, print_outputs = False):
        """
        Tests the model on given inputs.

        Parameters:
        inputs: 2D array of samples [count, sample]. If None, a single random sample is provided
        print_outputs (optional, default False): whether to print outputs info
        """
        if inputs is None:
            inputs = 2 * np.random.randint(0, 2, size=(1, self.input_size)) - 1
        elif len(inputs.shape) != 2:
            raise Exception("inputs shape should be 2D and follow pattern [inputs_count, input_vector]")

        outputs = self(inputs)
        if print_outputs:
            for idx in range(len(inputs)):
                if len(outputs[idx]) != 1:
                    print(f"Input:\n {inputs[idx]}\nOutputs:\n{outputs[idx][0]}\n{outputs[idx][1]}""")
                else:
                    print(f"Input:\n{inputs[idx]}\nOutput:\n{outputs[idx][0]}")
        return outputs

    def save(self):
        """
        Saves model to the saved_models directory.
        """
        path = get_file_path("saved_models", f"{self.name}_{time()}")
        np.save(path, self.hopfield_layer.weights)
        print(f"Model saved to: {path}.npy")

    @classmethod
    def load(cls, file_path: str):
        """
        Loads model from given file name.

        Parameters:
        file_path: path of the file with weights
        """
        weights = np.load(f"{file_path}")
        input_size = weights.shape[1]
        model = Model(input_size)
        model.hopfield_layer.weights = weights
        return model

    @classmethod
    def from_args(cls, arguments: Arguments):
        """
        Creates/Loads the model based on the command line arguments.

        Parameters:
        arguments: arguments from argument parser
        """
        model: cls
        if arguments.model_to_load != "":
            model = Model.load(get_file_path("saved_models", f"{arguments.model_to_load}.npy"))
        else:
            model = Model(arguments.input_size)
        model.name = arguments.model_name

        if arguments.patterns_file != "":
            patterns = Model.get_patterns(get_file_path("example_patterns", f"{arguments.patterns_file}.txt"))
            model.train(patterns)

        if arguments.save_model:
            model.save()

        return model
    
    @staticmethod
    def get_patterns(file_path):
        """
        Parses given file for patterns to use in training
        
        Parameters:
        file_path: path of the file with patterns
        """
        patterns: np.ndarray
        with open(f"{file_path}", "r") as file:
            pattern_strings = file.readlines()
            patterns = np.loadtxt(pattern_strings, delimiter=' ', dtype=np.int64)
        return patterns
