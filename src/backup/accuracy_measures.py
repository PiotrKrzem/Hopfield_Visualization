import numpy as np

# Function verifies if the prediction corresponds to the expected output
#
# expected_output - ground truth for the output
# output - predicted output
# 
# Returns: 1 if output is correct, 0 otherwise
def compute_accuracy_for_observation(expected_output: np.ndarray, output: np.ndarray) -> float:
    return 1 if np.argmax(expected_output) == np.argmax(output[-1]) else 0

# Function computes the number of correct predictions based on expected output
#
# expected_outputs - ground truth for the output
# outputs - predicted outputs
# 
# Returns: float accuracy
def compute_accuracy(expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
    true_values = [np.argmax(expected_outputs[idx]) == output for idx, output in enumerate(outputs)]

    return sum(true_values) / len(outputs)


# Function computes the loss for given epoch and sample
#
# expected_outputs - ground truth for the output for given sample
# outputs - predicted outputs in output layer
# 
# Returns: loss for selected sample
def compute_loss(expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
    output = outputs[-1]
    error = np.sum(expected_outputs*np.log(output+1e-9))

    return -error