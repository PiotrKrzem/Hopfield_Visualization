import sys, getopt

from src.domain.layer import LayerActivation
from src.data.loader import TrainingData

# ---------------------------- HELPER METHODS ----------------------------------

# Method retrieves activation function by name
#
# Parameters:
# name - name of the activation function
def get_activation_func_by_name(name: str) -> LayerActivation:
    if name == 'SIGMOID':
        return LayerActivation.SIGMOID
    elif name == 'SOFTMAX':
        return LayerActivation.SOFTMAX
    elif name == 'TANH':
        return LayerActivation.TANH
    elif name == 'RELU':
        return LayerActivation.RELU
    elif name == 'LINEAR':
        return LayerActivation.LINEAR
    
# Method retrieves data set by name
#
# Parameters:
# name - name of the data set
def get_dataset_by_name(name: str) -> TrainingData:
    if name == 'LENSES':
        return TrainingData.LENSES

class Arguments:
    def __init__(self) -> None:
        # -------------- DEFAULT PARAMETERS (IF NOT SPECIFIED BY THE USER) -----------
        self.hidden_layers_no = 2
        self.neurons_no = [16,16]
        self.activation_func = LayerActivation.RELU
        self.batch_size = 0
        self.epochs = 100
        self.learning_rate = 0.0001
        self.momentum = 0.1
        self.dataset = TrainingData.BANKNOTES
        self.early_stopping = 3

def parse_args():
    opts, _ = getopt.getopt(sys.argv[1:], "h:n:f:b:e:l:m:d:")

    arguments = Arguments()
    for opt, arg in opts:
        if opt == '-h':
            arguments.hidden_layers_no = int(arg) 
            print (f'Number of hidden layers: {arguments.hidden_layers_no}')
        elif opt == '-n':
            arguments.neurons_no = [int(x) for x in arg.split(',')]
            print (f'Number of neurons per layer: {arguments.neurons_no}')
        elif opt == '-f':
            arguments.activation_func = get_activation_func_by_name(arg)
            print (f'Activation function: {arguments.activation_func}')
        elif opt == '-b':
            arguments.batch_size = int(arg)
            print (f'Number of batches: {arguments.batch_size}')
        elif opt == '-e':
            arguments.epochs = int(arg)
            print (f'Number of epochs: {arguments.epochs}')
        elif opt == '-l':
            arguments.learning_rate = float(arg)
            print (f'Learning rate: {arguments.learning_rate}')
        elif opt == '-m':
            arguments.momentum = float(arg)
            print (f'Momentum: {arguments.momentum}')
        elif opt == '-d':
            arguments.dataset = get_dataset_by_name(arg)
            print (f'Dataset: {arg}')
        elif opt == '-s':
            arguments.dataset = int(arg)
            print (f'Early stopping: {arg}')

    assert(arguments.hidden_layers_no == len(arguments.neurons_no))
    return arguments