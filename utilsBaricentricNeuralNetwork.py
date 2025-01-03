import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras import layers, Model, activations
from tensorflow .keras.layers import Layer

# generate data using a function depending their domain and number of samples
def generate_training_data(f, x_range, num_samples):
    x = np.linspace(*x_range, num_samples)
    y = np.array([f(xi) for xi in x])
    return x, y


# the BaricentricNeuralNetwork: Pytorch
class BaricentricLayer(nn.Module):
    def __init__(self, points):
        super(BaricentricLayer, self).__init__()
        
        # Separate the input(x-coordinates and output values(y-values).
        self.x_coords = [p[0] for p in points]
        self.y_values = [p[1] for p in points]

        # Convert y-values to (fixed, non-trainable) parameters
        self.biases = nn.Parameter(torch.tensor(self.y_values, dtype=torch.float32), requires_grad=False)

    def forward(self, x):
        output = torch.zeros_like(x)  # Initialize output
        num_segments = len(self.x_coords) - 1  # Number of segments created by the points
        
        for i in range(num_segments):
            # Extract x_i, x_{i+1}, y_i, y_{i+1}
            x_i, x_next = self.x_coords[i], self.x_coords[i + 1]
            b_i, b_next = self.biases[i], self.biases[i + 1]
            
            # Barycentric coordinates t = (x - x_i) / (x_i+1 - x_i)
            t = (x - x_i) / (x_next - x_i)
            
            # Define contributions by segment
            step1 = (t >= 0).float()
            step2 = ((1 - t) > 0).float() #>= generate double value at exact x values
            relu1 = F.relu(1 - t)  # ReLU(1 - t)
            relu2 = F.relu(t)      # ReLU(t)
            
            # Output for this segment
            segment_output = step1 * relu1 * b_i + step2 * relu2 * b_next
            
            # Add the segment contribution to the total output
            output += segment_output

        return output


class BaricentricNetwork(nn.Module):
    def __init__(self, points):
        super(BaricentricNetwork, self).__init__()
        self.layer = BaricentricLayer(points)

    def forward(self, x):
        return self.layer(x)

# the BaricentricNeuralNetwork: Tensorflow
class BaricentricLayerTf(Layer):
    def __init__(self, points, **kwargs):
        super(BaricentricLayerTf, self).__init__(**kwargs)
        
        # Separate the input(x-coordinates and output values(y-values).
        self.x_coords = tf.constant([p[0] for p in points], dtype=tf.float32)
        self.y_values = tf.constant([p[1] for p in points], dtype=tf.float32)

    def call(self, x):
        output = tf.zeros_like(x)  # Initialize output
        num_segments = len(self.x_coords) - 1  # Number of segments created by the points
        
        for i in range(num_segments):
            # Extract x_i, x_{i+1}, y_i, y_{i+1}
            x_i, x_next = self.x_coords[i], self.x_coords[i + 1]
            b_i, b_next = self.y_values[i], self.y_values[i + 1]
            
            # Barycentric coordinates t = (x - x_i) / (x_i+1 - x_i)
            t = (x - x_i) / (x_next - x_i)
            
            # Define contributions by segment
            relu1 = activations.relu(1 - t)  # ReLU(1 - t)
            relu2 = activations.relu(t)      # ReLU(t)
            step1 = tf.cast(t >= 0, dtype=tf.float32)
            step2 = tf.cast((1 - t) > 0, dtype=tf.float32)
            
            # Output for this segment
            segment_output = step1 * relu1 * b_i + step2 * relu2 * b_next
            
            # Add the segment contribution to the total output
            output += segment_output

        return output


class BaricentricNetworkTf(Model):
    def __init__(self, points, **kwargs):
        super(BaricentricNetworkTf, self).__init__(**kwargs)
        self.layer = BaricentricLayerTf(points)

    def call(self, x):
        return self.layer(x)