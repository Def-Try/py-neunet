from neunet.NeuralNetwork import NeuralNetwork, load
import neunet.NeuralActivators as NeuralActivators
from neunet.NeuralInputs import NeuralInput

nn = load("test_nn.nn", activation=NeuralActivators.sigmoid, activation_derivative=NeuralActivators.sigmoid_derivative)

while True:
    inp = eval(input("Input: "))
    print("Prediction:", nn.predict([NeuralInput(inp)]))