import numpy as np

from src.domain.model import *
from src.domain.layer import *
from src.visualization import *

m = Model([
    InputLayer(4),
    Layer(16, LayerActivation.SOFTMAX),
    Layer(32, LayerActivation.SOFTMAX),
    Layer(16, LayerActivation.SOFTMAX),
    Layer(3, LayerActivation.SOFTMAX),
])

Visualization(m, np.random.rand(100,4), np.random.randint(0, 2, size=(100,3)))