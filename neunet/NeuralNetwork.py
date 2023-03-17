import numpy as np
import time
from neunet.__util__ import _progress_bar as progress_bar
from neunet.NeuralInputs import NeuralInput

"""Main Neural network class"""
class NeuralNetwork:
    """
    creates weights, biases matrix, sets up
    basic variables for network to be able to work

    KEYWORD ONLY! EVERY ARGUMENT IS REQUIRED!
    arg - input_size - Input layer size (int)
    arg - output_size - Output layer size (int)
    arg - learning_rate - Hyper parameter. By how much weights should be changed when
          training network. Google it, i am not explaining any further (number)
    arg - hidden_layers_count - Hidden layers count (int)
    arg - hidden_layers_width - Hidden layers width (int)
    arg - activation - Activation function (def)
    arg - activation_derivative - Actbstion derivative function (def)
    """
    def __init__(self, *_, input_size, output_size, learning_rate,
                 hidden_layers_count, hidden_layers_width, activation, activation_derivative):
        self.input_size = input_size
        self.hidden_layers_count = hidden_layers_count
        self.hidden_layers_widths = [hidden_layers_width] * hidden_layers_count
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + list(self.hidden_layers_widths) + [output_size]
        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.random.randn(layer_sizes[i+1]))

    """Internal. Processes input, predicts the answer."""
    def feed_forward(self, input_data):
        activations = [input_data]
        for i in range(self.hidden_layers_count + 1):
            x = np.dot(activations[i], self.weights[i]) + self.biases[i]
            activation = self.activation(x)
            activations.append(activation)
        return activations

    """
    trains neural network. i am not explaining how, just
    google it. thia is documentation, not guide or paper

    arg - inputs - Training inputs, that are fed
          to neural network (list of NeuralInput)
    arg - targets - Training outputs, that are
          targets for neural network to hit (list of NeuralInput)
    arg - epochs - Number of epochs to run training.
          More epochs = more accurate, but still can
          ruin entire network if too much. (int)
    return - Network accuracy, based on last epoch error.
    """
    def train(self, inputs, targets, *_, epochs=20000):
        real_inputs, real_targets = inputs.copy(), targets.copy()
        for i in range(len(inputs)):
            inputs[i] = inputs[i].get()
        for i in range(len(targets)):
            targets[i] = targets[i].get()
        train_start_time = time.time()
        print("Training neural network...")
        for epoch in range(epochs):
            print(progress_bar(epoch+1, epochs,eta=True, time_start=train_start_time),end=" ")
            err=np.array([])
            for input_data, target in zip(inputs, targets):
                activations = self.feed_forward(input_data)
                output_error = target - activations[-1]
                err = np.append(err, output_error)
                deltas = [output_error * self.activation_derivative(activations[-1])]
                for i in reversed(range(self.hidden_layers_count)):
                    delta = np.dot(deltas[-1], self.weights[i+1].T) * self.activation_derivative(activations[i+1])
                    deltas.append(delta)
                deltas.reverse()
                for i in range(self.hidden_layers_count + 1):
                    self.weights[i] += self.learning_rate * np.outer(activations[i], deltas[i])
                    self.biases[i] += self.learning_rate * deltas[i]
            print(f"err={round(sum(err)/len(err), 5)}", end="\r")
        predictions = self.predict(real_inputs)
        accuracy = 1 - (np.mean(targets) - np.mean(predictions))
        print()
        inputs = real_inputs
        targets = real_targets
        return accuracy

    """
    predict the answer based on input

    arg - input_data - neural network input data (list of NeuralInput)
    return - prediction (list of numbers)
    """
    def predict(self, input_data):
        for i in range(len(input_data)):
            input_data[i] = input_data[i].get()
        activations = self.feed_forward(input_data)
        predictions = activations[-1]
        return predictions

