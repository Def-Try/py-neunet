from neunet.NeuralNetwork import NeuralNetwork
import neunet.NeuralActivators as NeuralActivators
from neunet.NeuralInputs import NeuralInput

# Train data. This neural network has 3 inputs and 1 output,
# so arrays ahould be exact same sizes.
train_inputs = [NeuralInput([0,0,1]),
                NeuralInput([0,1,0]),
                NeuralInput([0,1,1]),
                NeuralInput([1,0,0]),
                NeuralInput([1,0,1]),
                NeuralInput([1,1,0]),
                NeuralInput([1,1,1])]
train_outputs = [NeuralInput([0]),
                NeuralInput([0]),
                NeuralInput([0]),
                NeuralInput([1]),
                NeuralInput([1]),
                NeuralInput([1]),
                NeuralInput([1])]

# if you didnt see it: if first item in input is 1, then
# output is 1. otherwise + 0

# create neural network
# structure:
#   H H H H H
# I H H H H H
# I H H H H H O
# I H H H H H
#   H H H H H
# type: fully connected
# activation function: sigmoid
nn = NeuralNetwork(input_size=3, output_size=1, learning_rate=0.1,
                   hidden_layers_count=5, hidden_layers_width=5,
                   activation=NeuralActivators.sigmoid,
                   activation_derivative=NeuralActivators.sigmoid_derivative)
# train and print accuracy
print("Accuracy:",nn.train(train_inputs, train_outputs, epochs=2000))

# take input from user. we are expecting bool-like int
# list, like "[1,0,0]". i am not implementing try-expect
# there because if user wanta to input something stupid,
# you shall make him able to do it
while True:
    inp = eval(input("Input: "))
    print("Prediction:", nn.predict([NeuralInput(inp)]))
