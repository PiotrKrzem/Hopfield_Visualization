import numpy as np
from sys import stdout
from typing import List
from random import randint
from colorsys import hsv_to_rgb
from dataclasses import dataclass


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from matplotlib.colors import ListedColormap
# from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import PathCollection

from src.domain.layer import Layer, InvisibleLayer, HopfieldLayer, OutputLayer
from src.domain.model import Model

from random import seed as r_seed
from numpy.random import seed as n_seed
r_seed(69), n_seed(69)

# Model rendering settings
MODEL_DEPTH_PX = 40
MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX = 20
MODEL_INVISIBLE_NEURONS_Y_OFFSET_PX = 1
# Line rendering settings
MULTIPLY_LINE_WIDTHS = 0.5
MINIMUM_LINE_WIDTH = 0.2
DEFAULT_LINE_COLOR = [31/256,119/256,180/256,1]
# Colors for Interpolate (HSV angle)
RED, YELLOW, GREEN = range(0, 180, 60)

def interpolate_color(values: np.ndarray):
    """ 
    Convert values in the range 0...1 to a color in the range start_color to stop_color.
    """
    startcolor = RED
    stopcolor = GREEN

    colors = np.zeros((values.shape[0], 3))
    hues = values * (stopcolor - startcolor) + startcolor
    for idx, hue in enumerate(hues):
        colors[idx, :] = hsv_to_rgb(hue/360, 1., 1.)

    return colors

def add_button(figure: Figure, x, y, w, h):
    def make_button_callback(figure: Figure, button: Button):
        def callback(event):
            if button.white:
                button.white = False
                button.color = 'red'
                button.hovercolor = 'red'
            else:
                button.white = True
                button.color = 'green'
                button.hovercolor = 'green'
            figure.canvas.draw()
        return callback

    button_axis = figure.add_axes([x,y,w,h])
    button = Button(button_axis, '', None, 'red', 'red')
    button.white = False # setattr(button, 'white', False)
    button.on_clicked(make_button_callback(figure, button))
    return button
@dataclass
class NeuronMarkersData:
    xyz: List[np.ndarray]
    layer: Layer
    marker: PathCollection

@dataclass
class ConnectionMarkerData:
    marker: Line3DCollection

class Visualization:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.update_idx = -1

        self.prepare_interface()
        self.prepare_train_test_buttons()
        self.prepare_markers()
        plt.show()

    def prepare_interface(self):
        self.figure = plt.figure("Visualization", figsize=(10,10))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.figure.canvas.mpl_connect('close_event', self.on_exit)

        self.axis:Axes = self.figure.add_subplot(111, projection='3d') # Axes3D
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

    def prepare_train_test_buttons(self):
        approx_side_length = np.sqrt(self.model.input_size)
        approx_side_length_incr = 1 if int(approx_side_length)*int(approx_side_length) < self.model.input_size else 0
        side_length = int(approx_side_length) + approx_side_length_incr

        self.input_vectors = []
        self.input_vector_buttons = [0]*self.model.input_size
        for idx in range(self.model.input_size):
            self.input_vector_buttons[idx] = add_button(self.figure,  0.05*(idx//side_length), 0.95 - 0.05*(idx%side_length), 0.05, 0.05)

        add_to_training_button_axis = self.figure.add_axes([0, 0.95 - 0.05*side_length, 0.05*side_length/2, 0.05])
        self.add_to_training_button = Button(add_to_training_button_axis, 'Add to train')
        self.add_to_training_button.on_clicked(self.add_to_training)

        training_button_axis = self.figure.add_axes([0.05*side_length/2, 0.95 - 0.05*side_length, 0.05*side_length/2, 0.05])
        self.training_button = Button(training_button_axis, 'Train')
        self.training_button.on_clicked(self.train_model)

        test_button_axis = self.figure.add_axes([0, 0.95 - 0.05*(side_length + 1), 0.05*side_length, 0.05])
        self.test_button = Button(test_button_axis, 'Test on pattern')
        self.test_button.on_clicked(self.test_model)

        self.output_vectors = []
        self.output_axes = []

    def prepare_markers(self):
        layer_width_div = MODEL_DEPTH_PX // (len(self.model.layers) - 1)
        self.layers_neurons_data: List[NeuronMarkersData] = []
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx == 0 or layer_idx == len(self.model.layers) - 1:
                approx_side_length = np.sqrt(layer.output_size)
                approx_side_length_incr = 1 if int(approx_side_length)*int(approx_side_length) < layer.output_size else 0
                side_length = int(approx_side_length) + approx_side_length_incr

                xz_start_value = -MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length // 2
                x_range = np.linspace(0, MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length, side_length) + xz_start_value
                z_range = np.linspace(0, MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length, side_length) + xz_start_value

                x_grid, z_grid = np.meshgrid(x_range, z_range)
                x_grid, z_grid = x_grid.flatten(), z_grid.flatten()
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)
            else:
                approx_side_length = np.sqrt(layer.output_size)
                approx_side_length_incr = 1 if int(approx_side_length)*int(approx_side_length) < layer.output_size else 0
                side_length = int(approx_side_length) + approx_side_length_incr
                xz_start_value = -MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length // 2
                x_range = np.linspace(0, MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length, side_length) + xz_start_value
                z_range = np.linspace(0, MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX * side_length, side_length) + xz_start_value
                x_grid, z_grid = np.meshgrid(x_range, z_range)
                x_grid, z_grid = x_grid.flatten(), z_grid.flatten()
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)

                if isinstance(layer, InvisibleLayer):
                    noise = np.arange(side_length)[:, np.newaxis] * MODEL_INVISIBLE_NEURONS_Y_OFFSET_PX
                    noise = np.tile(noise, [1, side_length]).flatten()[:x_grid.shape[0]]
                    y_grid += noise

            x_grid = x_grid[:layer.output_size]
            y_grid = y_grid[:layer.output_size]
            z_grid = z_grid[:layer.output_size]

            neuron_markers = self.axis.scatter(
                x_grid, y_grid, z_grid,
                marker='s', zorder = 2, s = 64,
                color='white', alpha=1,
                visible = not isinstance(layer, InvisibleLayer)
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

            if isinstance(dst_neurons_data.layer, InvisibleLayer) or isinstance(dst_neurons_data.layer, OutputLayer):
                src_neurons_data = self.layers_neurons_data[layer_idx - 1]
                dst_x, dst_y, dst_z = dst_neurons_data.xyz
                src_x, src_y, src_z = src_neurons_data.xyz
            else:
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
                linewidths = 0.4,
                color = DEFAULT_LINE_COLOR
            )
            self.layers_connections_data.append(
                ConnectionMarkerData(
                    connection_markers
                )
            )

        self.wrap_connections_data: List[ConnectionMarkerData] = []
        for layer_idx, dst_neurons_data in enumerate(self.layers_neurons_data):
            if not isinstance(dst_neurons_data.layer, InvisibleLayer) and not isinstance(dst_neurons_data.layer, HopfieldLayer):
                continue

            if isinstance(dst_neurons_data.layer, InvisibleLayer) :
                dst_x, dst_y, dst_z = dst_neurons_data.xyz
                src_x, src_y, src_z = dst_neurons_data.xyz
                offset_z = dst_z[-1] - dst_z[0] + MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX
                src_z = src_z.copy()
                dst_z = dst_z.copy() - offset_z
                
            elif isinstance(dst_neurons_data.layer, HopfieldLayer):
                new_dst_neurons_data = self.layers_neurons_data[layer_idx + 1]
                src_neurons_data = self.layers_neurons_data[layer_idx - 1]
                dst_x, dst_y, dst_z = new_dst_neurons_data.xyz
                src_x, src_y, src_z = src_neurons_data.xyz
                offset_z = dst_z[-1] - dst_z[0] + MODEL_VISIBLE_NEURONS_XZ_DISTANCE_PX
                src_z = src_z.copy() - offset_z
                dst_z = dst_z.copy() - offset_z

            delta_x = dst_x - src_x
            delta_y = dst_y - src_y
            delta_z = dst_z - src_z

            wrap_connection_markers = self.axis.quiver(
                src_x, src_y, src_z,
                delta_x, delta_y, delta_z,
                arrow_length_ratio = 0,
                linewidths = 0.4
            )
            self.wrap_connections_data.append(
                ConnectionMarkerData(
                    wrap_connection_markers
                )
            )

    def update(self):
        if self.update_idx == -1:
            print("No vector is loaded. Use 'z' to generate a new one.")
            return

        # Values for update_idx are hardcoded since the amount of layers is static (5).
        # Update idx determines which part of the network is visually updated
        # 
        # 0  1   2+6i 3+6i 4+6i 5+6i 6+6i
        # o --- x --- o --- x --- o
        #  7+6i |           | 5+6 
        #       -------------
        #            6+6i

        for data in self.layers_connections_data:
            data.marker.set_color(DEFAULT_LINE_COLOR)
            data.marker.set_linewidth(0.4)
        for data in self.wrap_connections_data:
            data.marker.set_color(DEFAULT_LINE_COLOR)
            data.marker.set_linewidth(0.4)

        # Before cycle
        if self.update_idx < 2:
            layer = self.model.layers[0]
            output_size = layer.output_size

            if self.update_idx == 1:
                self.input = layer(self.input)

            normalized_output = self.input - self.input.min()
            normalized_output = normalized_output / (normalized_output.max()+1.0e-9)

            color = interpolate_color(normalized_output)
            alpha = np.ones((output_size, 1))
            color_rgba = np.hstack((color, alpha))
            if self.update_idx == 0:
                self.layers_neurons_data[0].marker.set_color(color_rgba)
            else:
                widths = normalized_output
                self.layers_connections_data[0].marker.set_color(color_rgba)
                self.layers_connections_data[0].marker.set_linewidth(MULTIPLY_LINE_WIDTHS * widths + MINIMUM_LINE_WIDTH)
        # Cycle
        else:
            if self.update_idx % 6 == 2:
                layer_idx = 1
            elif self.update_idx % 6 == 3:
                layer_idx = 2
            elif self.update_idx % 6 == 4:
                layer_idx = 2
            elif self.update_idx % 6 == 5:
                layer_idx = 3
            elif self.update_idx % 6 == 0:
                layer_idx = 4
            else: # == 1
                layer_idx = 0

            layer = self.model.layers[layer_idx]
            output_size = layer.output_size

            if self.update_idx % 6 == 3:
                self.input = layer(self.input)

            normalized_output = self.input - self.input.min()
            normalized_output = normalized_output / (normalized_output.max()+1.0e-9)

            color = interpolate_color(normalized_output)
            alpha = np.ones((output_size, 1))
            color_rgba = np.hstack((color, alpha))

            # Update loop below the structure (before main structure since it can edit color_rgba)
            if self.update_idx % 6 in [5,0,1]:
                if self.update_idx % 6 == 5:
                    wrap_idx = 2
                elif self.update_idx % 6 == 0:
                    wrap_idx = 1
                else: # == 1
                    wrap_idx = 0
                widths = normalized_output
                self.wrap_connections_data[wrap_idx].marker.set_color(color_rgba)
                self.wrap_connections_data[wrap_idx].marker.set_linewidth(MULTIPLY_LINE_WIDTHS * widths + MINIMUM_LINE_WIDTH) # color == widths in this case

            # Update main structure
            if self.update_idx % 6 == 3:
                self.layers_neurons_data[layer_idx].marker.set_color(color_rgba)
            elif self.update_idx % 6 == 0:
                self.layers_neurons_data[layer_idx].marker.set_color(color_rgba)
                self.draw_output()
            elif self.update_idx % 6 != 1: 
                next_layer_output_size = self.model.layers[layer_idx + 1].output_size
                if self.update_idx % 6 == 2:
                    color_rgba = np.tile(color_rgba, (1, 1, next_layer_output_size)).reshape(output_size * next_layer_output_size, 4)
                    widths = np.tile(normalized_output[:,np.newaxis], (1, next_layer_output_size)).reshape(output_size * next_layer_output_size)
                else:
                    widths = normalized_output
                self.layers_connections_data[layer_idx].marker.set_color(color_rgba) # NOT layer_idx + 1 because this array's size is one less
                self.layers_connections_data[layer_idx].marker.set_linewidth(MULTIPLY_LINE_WIDTHS * widths + MINIMUM_LINE_WIDTH)

        self.update_idx += 1
        self.figure.canvas.draw()

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'z':
            self.input = 2 * np.random.randint(0, 2, size=(self.model.input_size)) - 1
            self.update_idx = 0
            self.update()
        elif event.key == 'x':
            self.update()

    def on_mouse_press(self, event):
        return

    def on_exit(self, event):
        return

    def add_to_training(self, event):
        input_vector = np.zeros((self.model.input_size))
        for idx, btn in enumerate(self.input_vector_buttons):
            input_vector[idx] = 1 if btn.white else -1
            btn.white = False
            btn.color = 'red'
            btn.hovercolor = 'red'

        self.input_vectors.append(input_vector)
        self.figure.canvas.draw()
        print(f"Added vector to training set:\n{input_vector}")

    def train_model(self, event):
        input_vectors = np.array(self.input_vectors)
        for axis in self.output_axes:
            axis.remove()
        self.output_axes.clear()
        self.output_vectors.clear()
        self.model.train(input_vectors)
        self.input_vectors.clear()
        print(f"Model has been trained")
        self.update_idx = 0
        self.clean_model()

    def test_model(self, event):
        input_vector = np.zeros((self.model.input_size))
        for idx, btn in enumerate(self.input_vector_buttons):
            input_vector[idx] = 1 if btn.white else -1

        print(f"Test on:\n{input_vector}")
        self.input = input_vector
        self.update_idx = 0
        self.clean_model()
        self.update()

    def draw_output(self):
        normalized_output:np.ndarray = self.input - self.input.min()
        normalized_output = normalized_output / (normalized_output.max() + 1.0e-9)
        reshape_to = int(np.ceil(np.sqrt(normalized_output.shape[0])))
        for vector in self.output_vectors:
            if np.array_equal(vector, self.input):
                return

        image_axis = self.figure.add_axes([0+0.2*len(self.output_axes),0,0.2,0.2])
        img = normalized_output.reshape(reshape_to, reshape_to)
        image_axis.matshow(np.rot90(img, 3), cmap = ListedColormap(['red','green']))
        self.output_vectors.append(self.input.copy())
        self.output_axes.append(image_axis)

    def clean_model(self):
        for data in self.layers_neurons_data:
            data.marker.set_color('white')
        for data in self.layers_connections_data:
            data.marker.set_color(DEFAULT_LINE_COLOR)
            data.marker.set_linewidth(0.4)
        for data in self.wrap_connections_data:
            data.marker.set_color(DEFAULT_LINE_COLOR)
            data.marker.set_linewidth(0.4)
        self.figure.canvas.draw()