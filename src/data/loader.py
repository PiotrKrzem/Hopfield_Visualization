import numpy as np
from os import path
from enum import Enum
from typing import Tuple
from sklearn.model_selection import train_test_split


DIR_NAME = 'downloads'
LENSES_CLASSES = 3
CARS_CLASSES = 4
BANKNOTES_CLASSES = 2
DIGITS_CLASSES = 10


def get_file_path(file_name: str) -> str:
    file_absolute_path = path.dirname(__file__)
    return path.join(file_absolute_path, DIR_NAME, file_name)

def get_output_for_row(label: int, classes_no: int) -> np.ndarray:
    output = np.zeros(classes_no)
    output[int(label) - 1] = 1

    return output

def import_lenses_data() -> Tuple[int, int, np.ndarray, np.ndarray]:
    file_path = get_file_path("lenses.data")

    with open(file_path) as file:
        lines = (line for line in file)
        data = np.loadtxt(lines, delimiter=' ', dtype=np.int64)[:,1:]

        inputs = data[:, :-1]
        inputs = inputs / (inputs.max(axis = 0) + 0.001)
        outputs = np.array([get_output_for_row(label, LENSES_CLASSES) for label in data[:, -1]])

        return len(inputs[0]), LENSES_CLASSES, inputs, outputs
    
def import_cars_data() -> Tuple[int, int, np.ndarray, np.ndarray]:
    file_path = get_file_path("car.data")

    with open(file_path) as file:
        lines = []
        for line in file.readlines():
            line = line.replace("unacc","0")
            line = line.replace("acc","1")
            line = line.replace("vgood","3")
            line = line.replace("good","2")

            line = line.replace("low","0")
            line = line.replace("small","0")
            line = line.replace("med","1")
            line = line.replace("vhigh","3")
            line = line.replace("high","2")
            line = line.replace("big","2")

            line = line.replace("5more","5")
            line = line.replace("more","5")
            lines.append(line)

        data = np.loadtxt(lines, delimiter=',', dtype=np.int64)

        inputs = data[:, :-1]
        inputs = inputs / (inputs.max(axis = 0) + 0.001)
        outputs = np.array([get_output_for_row(label, CARS_CLASSES) for label in data[:, -1]])

        return len(inputs[0]), CARS_CLASSES, inputs, outputs

def import_banknotes_data() -> Tuple[int, int, np.ndarray, np.ndarray]:
    file_path = get_file_path("data_banknote_authentication.txt")

    with open(file_path) as file:
        lines = (line for line in file)
        data = np.loadtxt(lines, delimiter=',', dtype=np.float32)

        inputs = data[:, :-1]
        inputs = inputs / (inputs.max(axis = 0) + 0.001)
        outputs = np.array([get_output_for_row(label, BANKNOTES_CLASSES) for label in data[:, -1]])

        return len(inputs[0]), BANKNOTES_CLASSES, inputs, outputs
    
def import_digits_data() -> Tuple[int, int, np.ndarray, np.ndarray]:
    file_path = get_file_path("optdigits.tra")

    with open(file_path) as file:
        lines = (line for line in file)
        data = np.loadtxt(lines, delimiter=',', dtype=np.float32)

        inputs = data[:, :-1]
        inputs = inputs / (inputs.max(axis = 0) + 0.001)
        outputs = np.array([get_output_for_row(label, DIGITS_CLASSES) for label in data[:, -1]])

        return len(inputs[0]), DIGITS_CLASSES, inputs, outputs

class TrainingData(Enum):
    LENSES = import_lenses_data()
    CARS = import_cars_data()
    BANKNOTES = import_banknotes_data()
    DIGITS = import_digits_data()

def train_val_test_split(X, Y, train_ratio = 0.8, validation_ratio = 0.1, test_ratio = 0.1):
    assert(train_ratio + validation_ratio + test_ratio == 1.0)

    # train is now 75% of the entire data set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 - train_ratio)

    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 

    return x_train, y_train, x_val, y_val, x_test, y_test

if __name__ == "__main__":
    print(TrainingData.LENSES.value)
    print(TrainingData.BANKNOTES.value)
    print(TrainingData.CARS.value)
