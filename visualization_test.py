import numpy as np

from src.domain.model import Model
from src.visualization import Visualization

input_size = 16

m = Model(input_size)
Visualization(m, np.random.rand(1,input_size), 2 * np.random.randint(0, 2, size=(1,input_size) - 1))
