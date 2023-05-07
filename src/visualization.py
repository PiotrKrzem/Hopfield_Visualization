import numpy as np
from sys import stdout
from typing import List
from random import randint
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import PathCollection

from src.domain.layer import Layer
from src.domain.model import Model

from random import seed as r_seed
from numpy.random import seed as n_seed
r_seed(69), n_seed(69)

MODEL_DEPTH_PX = 40
MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER = 5
MODEL_NEURONS_DISTANCE_PX = 20
NOISE_AMOUNT = MODEL_NEURONS_DISTANCE_PX // 10
NOISE_AMOUNT = 2
SEAPARATE_FIRST_LAST_LAYER = 2
@dataclass
class NeuronMarkersData:
    xyz: List[np.ndarray]
    layer: Layer
    marker: PathCollection

@dataclass
class ConnectionMarkerData:
    layer: Layer
    marker: Line3DCollection

class Visualization:
    def __init__(self, model: Model, inputs:np.ndarray, outputs: np.ndarray) -> None:
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

        self.prepare_interface()
        self.prepare_markers()
        plt.show()

    def prepare_interface(self):
        self.figure = plt.figure("Visualization", figsize=(10,10))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.figure.canvas.mpl_connect('close_event', self.on_exit)

        self.axis:Axes = self.figure.add_subplot(111, projection='3d')
        self.axis.spines['bottom'].set(color = 'white', linewidth = 0)
        self.axis.spines['top'].set(color = 'white', linewidth = 0)
        self.axis.spines['left'].set(color = 'white', linewidth = 0)
        self.axis.spines['right'].set(color = 'white', linewidth = 0)
        self.axis.grid(False)
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_zticks([])
        self.axis.set_axis_off()
        self.axis.set_facecolor([0.1,0.1,0.1])
        self.figure.set_facecolor([0.1,0.1,0.1])

    def prepare_markers(self):
        layer_width_div = MODEL_DEPTH_PX // (len(self.model.layers) - 1)

        self.layers_neurons_data: List[NeuronMarkersData] = []
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx == 0 or layer_idx == len(self.model.layers) - 1:
                side_length = layer.output_size
                x_start_value = int(-MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER * side_length / 2)
                x_grid = np.linspace(0, MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER * side_length, side_length) + x_start_value
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)
                y_grid += -layer_width_div * SEAPARATE_FIRST_LAST_LAYER if not layer_idx else layer_width_div * SEAPARATE_FIRST_LAST_LAYER
                z_grid = np.zeros((x_grid.shape[0]))
            else:
                approx_side_length = np.sqrt(layer.output_size)
                approx_side_length_incr = 1 if int(approx_side_length)*int(approx_side_length) < layer.output_size else 0
                side_length = int(approx_side_length) + approx_side_length_incr
                xz_start_value = -MODEL_NEURONS_DISTANCE_PX * side_length // 2
                x_range = np.linspace(0, MODEL_NEURONS_DISTANCE_PX * side_length, side_length) + xz_start_value
                z_range = np.linspace(0, MODEL_NEURONS_DISTANCE_PX * side_length, side_length) + xz_start_value
                x_grid, z_grid = np.meshgrid(x_range, z_range)
                x_grid, z_grid = x_grid.flatten(), z_grid.flatten()
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)

                x_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                y_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                z_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                x_grid += x_noise
                y_grid += y_noise
                z_grid += z_noise

            x_grid = x_grid[:layer.output_size]
            y_grid = y_grid[:layer.output_size]
            z_grid = z_grid[:layer.output_size]

            neuron_markers = self.axis.scatter(
                x_grid, y_grid, z_grid,
                marker='s', zorder = 2, s = 64,
                color='white', alpha=1
            )
            self.layers_neurons_data.append(
                NeuronMarkersData(
                    (x_grid, y_grid, z_grid),
                    layer,
                    neuron_markers
                )
            )

        self.layers_connections_data: List[ConnectionMarkerData] = []
        for layer_idx, dst_neurons_data in enumerate(self.layers_neurons_data):
            if not layer_idx: continue

            x,y,z = dst_neurons_data.xyz
            src_neurons_data = self.layers_neurons_data[layer_idx - 1]
            dst_x = np.tile(x, src_neurons_data.layer.output_size)
            dst_y = np.tile(y, src_neurons_data.layer.output_size)
            dst_z = np.tile(z, src_neurons_data.layer.output_size)

            tile_amount = x.shape[0]
            x,y,z = src_neurons_data.xyz
            src_x = np.tile(x[np.newaxis].T, (1,tile_amount)).flatten()
            src_y = np.tile(y[np.newaxis].T, (1,tile_amount)).flatten()
            src_z = np.tile(z[np.newaxis].T, (1,tile_amount)).flatten()

            delta_x = dst_x - src_x
            delta_y = dst_y - src_y
            delta_z = dst_z - src_z

            connection_markers = self.axis.quiver(
                src_x, src_y, src_z,
                delta_x, delta_y, delta_z,
                arrow_length_ratio = 0,
                linewidths = 0.4
            )
            self.layers_connections_data.append(
                ConnectionMarkerData(
                    layer,
                    connection_markers
                )
            )

    def update(self):
        idx = randint(0, self.inputs.shape[0] - 1)
        output = self.inputs[idx]
        expected_output = self.outputs[idx]
        print(f"Input data: \n{output}\nExpected output: \n{expected_output}")

        for layer_idx, layer in enumerate(self.model.layers):
            output_size = layer.output_size
            # output = np.random.rand(output_size) #TODO REMOVE and activate stuff below
            if layer_idx:
                output = layer(output)

            normalized_output = output - output.min()
            normalized_output = normalized_output / normalized_output.max()

            # if layer_idx == 0 or layer_idx == len(self.model.layers) - 1:
            color = np.tile(np.array([[1,1,1]]), (output_size, 1)) * np.tile(normalized_output[np.newaxis].T, 3)
            # else:
            #     color = np.tile(np.array([[0,0,1]]), (output_size, 1)) * np.tile(output[np.newaxis].T, 3)
            # color = np.maximum(color, 0.5)
            alpha = np.ones((output_size, 1))
            color_rgba = np.hstack((color, alpha))
            self.layers_neurons_data[layer_idx].marker.set_color(color_rgba)

            # weights = layer.weights.flatten()
            # color = np.tile(np.array([[0,0,1]]), (weights_size, 1)) * np.tile(weights[np.newaxis].T, 3)
            # alpha = np.ones((weights_size, 1))
            # color_rgba = np.hstack((color, alpha))
            # self.layers_connections_data[layer_idx - 1].marker.set_color(color_rgba)

            if layer_idx == len(self.model.layers) - 1: break

            next_layer_output_size = self.model.layers[layer_idx + 1].output_size
            color_rgba = np.tile(color_rgba, (1,1,next_layer_output_size)).reshape(output_size * next_layer_output_size,4)
            widths = np.tile(normalized_output, (1,next_layer_output_size)).reshape(output_size * next_layer_output_size)
            self.layers_connections_data[layer_idx].marker.set_color(color_rgba) # NOT layer_idx + 1 because this array's size is one less
            self.layers_connections_data[layer_idx].marker.set_linewidth(widths)

        print(f"Output: \n{output}")
        self.figure.canvas.draw()

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'x':
            self.update()

    def on_mouse_press(self, event):
        return

    def on_exit(self, event):
        return