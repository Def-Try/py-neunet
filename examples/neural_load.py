from neunet.NeuralNetwork import NeuralNetwork, load
import neunet.NeuralActivators as NeuralActivators
from neunet.NeuralInputs import NeuralInput

# load perceptron neural network from previous example
nn = load("test_nn.nn", activation=NeuralActivators.sigmoid, 
                        activation_derivative=NeuralActivators.sigmoid_derivative)

# and take input from user
while True:
    inp = eval(input("Input: "))
    print("Prediction:", nn.predict([NeuralInput(inp)]))