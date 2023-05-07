import sys, getopt
import numpy as np

class Arguments:
    def __init__(self) -> None:
        # -------------- Default parameters -----------
        self.model_to_load = ""
        self.input_size = 16
        self.patterns_file = ""
        self.save_model = False
        self.model_name = "model"

def parse_args():
    """
    Parses command line arguments.
    
    Available arguments:
    -m: name of the npy file with model weights inside saved_models folder. If not provided, model is generated from input_size
    -i: input size of the model. Used only when model file is not provided. Defaults to 16
    -t: name of the txt file with patterns inside example_patterns folder. If not provided, model is not trained (random initialization)
    -s: whether to save output model. Defaults to False
    -n: name of the npy file to save the model to. Defaults to 'model'
    """
    opts, _ = getopt.getopt(sys.argv[1:], "m:s:n:i:t:")

    arguments = Arguments()
    for opt, arg in opts:
        if opt == '-m':
            arguments.model_to_load = str(arg) 
        elif opt == '-i':
            arguments.input_size = int(arg)
        elif opt == '-t':
            arguments.patterns_file = str(arg)
        elif opt == '-s':
            arguments.save_model = bool(arg) 
        elif opt == '-n':
            arguments.model_name = str(arg) 

    if arguments.model_to_load != "":
        print(f'Model to load: {arguments.model_to_load}')
    else:
        print(f'Model input size: {arguments.input_size}')

    if arguments.save_model:
        print(f'Model will be saved to: saved_models/{arguments.model_name}.npy')
    else:
        print(f"Model will not be saved")

    if arguments.patterns_file != "":
        print(f"Model will be trained with patterns from example_patterns/{arguments.patterns_file}.txt")
    else:
        print(f"Model will not be trained")
    return arguments

