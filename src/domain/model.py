import numpy as np
from tqdm import tqdm
from signal import signal, SIGINT
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.domain.layer import Layer, InputLayer
from  src.accuracy_measures import compute_accuracy, compute_loss, compute_accuracy_for_observation

# ----------------------- HELPER METHODS ------------------------------

# Method divides a given input set into batches of the corresponding size
#
# Parameters:
# inputs     - ndarray input to be divided into batches
# batch_size - desired size of a single batch
#
# Returns: ndarray of object batches 
# (i.e. 
# batch[i][0] - index of input in original array,
# batch[i][1] - input data)
def divide_data_to_batches(inputs: np.ndarray, batch_size: int) -> np.ndarray:
    batch_no = np.arange(batch_size, len(inputs), batch_size)               
    inputs_random = np.array(list(enumerate(inputs.copy())), dtype=object)   
    np.random.shuffle(inputs_random)

    return np.split(inputs_random, batch_no)

def plot_accuracy_loss(accuracy: List[float], loss:List[float], val_accuracy:List[float], val_loss:List[float]):
    epochs = list(range(len(accuracy)))

    fig = plt.figure("Accuracy & Loss statistics")
    ax_acc = fig.add_subplot(1, 2, 1)
    ax_loss = fig.add_subplot(1, 2, 2)
    ax_acc.plot(epochs, accuracy, '-b', label="Train Accuracy")
    ax_acc.plot(epochs, val_accuracy, '-r', label="Validation Accuracy")
    ax_acc.legend(loc = 'upper left', frameon=False)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")

    ax_loss.plot(epochs, loss, '-b', label="Train Loss")
    ax_loss.plot(epochs, val_loss, '-r', label="Validation Loss")
    ax_loss.legend(loc = 'upper left', frameon=False)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")

    fig.show()

# -------------------------- NETWORK MODEL ------------------------------

class Model:
    # Method initialize the network model
    #
    # Parameters:
    # layers     - layers with which the model is constructed
    # input_size - size of the input data
    def __init__(self, layers: List[Layer]) -> None:
        assert(type(layers[0]) is InputLayer)
        self.layers = layers

        self.stop_training = False
        signal(SIGINT, self.stop)

        for idx, layer in enumerate(self.layers):
            if not idx: continue
            layer.__construct__(layers[idx - 1].output_size)

    # Method performs forward propagation
    #
    # Parameters:
    # input_row - input data sent to the model
    def __forward__(self, input_row: np.ndarray) -> np.ndarray:
        input = input_row
        outputs = [input_row]                   # outputs stores corresponding outputs of each layer

        for layer in self.layers:
            input = layer(input)       # activating neurons for given input
            outputs.append(input)               # appending the output of the given layer

        return outputs

    # Method performs backward propagation in model training
    #
    # Parameters:
    # outputs            - outputs of the output layer obtained after forward propagation
    # expected_output    - output that was expected for given input data
    # learn_rate         - learning rate parameter of the model
    # momentum           - momentum parameter of the model
    # prev_weights_diffs - amount by which the weights were updated in previous iteration (used along with 
    #                      momentum)
    # is_stochastic      - flag indicating if stochastic training is performed (i.e. if the weight are to 
    #                      be updated for each sample)
    def __backward__(self, 
                     outputs: List[List[int]], 
                     expected_output: np.ndarray, 
                     learn_rate: float, 
                     momentum: float,
                     prev_weights_diffs: List[np.ndarray],
                     is_stochastic: bool) -> Tuple[List[np.ndarray],List[np.ndarray]]:
        error = None                # initial prediction error
        weights_diffs = []          # structure storing differences applied to weights per layer
        bias_diffs = []             # structure storing differences applied to biases per layer

        # iterating over all layers in reversed order (i.e. starting from the output layer)
        for idx, layer in reversed(list(enumerate(self.layers))):
            activation = layer.activation       # activation function assigned to layer
            output = outputs[idx + 1]           # output of the given layer
            input = outputs[idx]                # input received by a layer

            # if weights were previously adjusted then take corresponding differences for layer
            prev_weights_diff = prev_weights_diffs[idx] if prev_weights_diffs else 0

            if error is None:
                error = (output - expected_output)                   # error computed for output layer
            else:
                error = np.dot(error, self.layers[idx + 1].weights)  # error computed for hidden layers

            # multiplying error by derivative of activation (for output produced by the layer)
            delta = error * activation(output, der=True)
        
            # computing weight and bias differences (Kronecker product is used as the sizes of matrices
            # have different dimensions)
            weights_diff = learn_rate * np.kron(delta, input) + prev_weights_diff * momentum
            bias_diff = learn_rate * delta

            # if weights are to be updated for each sample then updating them
            if is_stochastic == True:
                layer.__update_weights__(weights_diff, bias_diff)

            # appending computed weights differences to data structures returned by the method
            weights_diffs.append(weights_diff)
            bias_diffs.append(bias_diff)

        # reversing order in structures to simplify computations in next iteration of training as back propagation proceeds from the last layer
        return weights_diffs[::-1], bias_diffs[::-1]
    

    # Method trains the model in a stochastic manner (meaning that the data is not divided into batches but considered as one and the weights are updated for each sample)
    #
    # Parameters:
    # epochs           - number of epochs used in the iterations
    # inputs           - input based on which the model is trained
    # expected_outputs - output that is expected for given input
    # learn_rate       - learning rate of the model
    # momentum         - momentum of the model
    def __train_stochastic__(self, 
                             epochs: int, 
                             inputs: np.ndarray, 
                             expected_outputs: np.ndarray, 
                             val_inputs: np.ndarray,
                             val_expected_outputs: np.ndarray,
                             learn_rate: float, 
                             momentum: float,
                             early_stopping_threshold: int) -> None:
        
        early_stopping = early_stopping_threshold
        previous_accuracy = 0
        accuracy_per_epoch = []
        val_accuracy_per_epoch = []
        loss_per_epoch = []
        val_loss_per_epoch = []

        # iterating based on the number of epochs
        for epoch in range(epochs):
            if self.stop_training: break
            total_loss = 0          # total loss for a given epoch
            results = []            # results obtained for epoch (used to compute the accuracy)
            weight_diffs = None     # differences in weights for given iteration

            # iterating over all samples
            for idx in range(len(inputs)):
                inputs_sample = inputs[idx]         
                expected_output = expected_outputs[idx]

                # getting predicted output based of forward propagation
                outputs = self.__forward__(inputs_sample)
                # updating weights in backward propagation
                weight_diffs, _ = self.__backward__(outputs, expected_output, learn_rate, momentum, weight_diffs, is_stochastic=True)
                
                # computing total loss
                total_loss += compute_loss(expected_output, outputs)
                # appending results for a given output (used in computation of accuracy)
                results.append(np.argmax(outputs[-1]))

            # computing accuracy and printing results of the iteration
            accuracy = compute_accuracy(expected_outputs, results)
            total_loss /= inputs.shape[0]
            if accuracy <= previous_accuracy:
                early_stopping -= 1
            else:
                early_stopping = early_stopping_threshold
            previous_accuracy = accuracy

            val_accuracy, val_loss = self.__test__(val_inputs, val_expected_outputs)
            accuracy_per_epoch.append(accuracy)
            val_accuracy_per_epoch.append(val_accuracy)
            loss_per_epoch.append(total_loss)
            val_loss_per_epoch.append(val_loss)

            print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {total_loss}, Val_Accuracy: {val_accuracy}, Val_Loss: {val_loss}")
            if early_stopping == 0:
                print(f"-- Early stopping --")
                break
        plot_accuracy_loss(accuracy_per_epoch, loss_per_epoch, val_accuracy_per_epoch, val_loss_per_epoch)


    # Method trains the model in a mini-batch manner (meaning that the data is divided into batches and weights are updated after a given batch is processed)
    #
    # Parameters:
    # epochs           - number of epochs used in the iterations
    # inputs           - input based on which the model is trained
    # expected_outputs - output that is expected for given input
    # learn_rate       - learning rate of the model
    # momentum         - momentum of the model
    # batch_size       - size of the batches into which the data is divided
    def __train_batches__(self, 
                          epochs: int, 
                          inputs: np.ndarray, 
                          expected_outputs: np.ndarray, 
                          val_inputs: np.ndarray,
                          val_expected_outputs: np.ndarray,
                          learn_rate: float, 
                          momentum: float,
                          early_stopping_threshold: int,
                          batch_size: int) -> None:
        # dividing data into batches
        batches = divide_data_to_batches(inputs, batch_size)

        early_stopping = early_stopping_threshold
        previous_accuracy = 0
        accuracy_per_epoch = []
        val_accuracy_per_epoch = []
        loss_per_epoch = []
        val_loss_per_epoch = []

        # iterating based on the number of epochs
        for epoch in range(epochs):
            if self.stop_training: break
            total_loss = 0                  # total loss for a given iteration
            correct_predictions = 0         # number of correct predictions (used in the accuracy)

            # iterating over all batches
            for batch in batches:
                # initializing sums used to update weights after a batch is fully processed
                weight_diff_sum = None
                bias_diff_sum = None
                weight_diffs = None
                
                # iterating over all samples
                for idx in range(len(batch)):
                    inputs_sample = batch[idx][1]
                    expected_output = expected_outputs[batch[idx][0]]

                    # similar computations to the stochastic method
                    outputs = self.__forward__(inputs_sample)
                    weight_diffs, bias_diffs = self.__backward__(outputs, expected_output, learn_rate, momentum, weight_diffs, is_stochastic=False)

                    # updating weight and bias differences sums
                    weight_diff_sum = weight_diffs if weight_diff_sum is None else [weight_diff_sum[idx] + diffs for idx, diffs in enumerate(weight_diffs)]
                
                    bias_diff_sum = bias_diffs if bias_diff_sum is None else [bias_diff_sum[idx] + diffs for idx, diffs in enumerate(bias_diffs)]

                    # computing prediction evaluation meassures
                    correct_predictions += compute_accuracy_for_observation(expected_output, outputs)
                    total_loss += compute_loss(expected_output, outputs)
                
                # updating weights for each layer after given batch is processed
                for idx, layer in reversed(list(enumerate(self.layers))):
                    layer.__update_weights__(weight_diff_sum[idx], bias_diff_sum[idx])

            # computing accuracy and printing result of the iteration
            accuracy = correct_predictions / len(expected_outputs)
            total_loss /= inputs.shape[0]
            if accuracy <= previous_accuracy:
                early_stopping -= 1
            else:
                early_stopping = early_stopping_threshold
            previous_accuracy = accuracy

            val_accuracy, val_loss = self.__test__(val_inputs, val_expected_outputs)
            accuracy_per_epoch.append(accuracy)
            val_accuracy_per_epoch.append(val_accuracy)
            loss_per_epoch.append(total_loss)
            val_loss_per_epoch.append(val_loss)

            print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {total_loss}, Val_Accuracy: {val_accuracy}, Val_Loss: {val_loss}")
            if early_stopping == 0:
                print(f"-- Early stopping --")
                break
        plot_accuracy_loss(accuracy_per_epoch, loss_per_epoch, val_accuracy_per_epoch, val_loss_per_epoch)

    def __test__(self, inputs: np.ndarray, expected_outputs: np.ndarray, print_stats = False):
        total_loss = 0
        correct_predictions = 0

        # iterating over all samples
        for idx in range(len(inputs)):
            inputs_sample = inputs[idx]         
            expected_output = expected_outputs[idx]

            # getting predicted output based of forward propagation
            outputs = self.__forward__(inputs_sample)
            
            # computing total loss, checking predictions
            total_loss += compute_loss(expected_output, outputs)
            correct_predictions += compute_accuracy_for_observation(expected_output, outputs)

        accuracy = correct_predictions / len(expected_outputs)
        total_loss /= inputs.shape[0]

        if print_stats:
            print(f"Test results - Accuracy: {accuracy}, Loss: {total_loss}")
        return accuracy, total_loss
        
    def stop(self, *args):
        if self.stop_training: exit(0)
        self.stop_training = True