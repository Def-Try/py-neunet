# Py-NeuNet

## neunet.NeuralNetwork
Provides base neural network capabilities.
### def load
  loads neural network from file
  load(path, *_, activation, activation_derivative) -> neural_network
    path - path from where to load neural network (string)
    neural_network - loaded neural network (neunet.NeuralNetwork.NeuralNetwork)
    activation - Activation function (def)
    activation_derivative - Activation derivative function (def)
### class neunet.NeuralNetwork.NeuralNetwork
  Actual neural network class.
  \__init\__(self, *_, input_size, output_size, learning_rate, hidden_layers_count, 
           hidden_layers_width, activation, activation_derivative)
    creates weights, biases matrix, sets up
    basic variables for network to be able to work

    input_size - Input layer size (int)
    output_size - Output layer size (int)
    learning_rate - Hyper parameter. By how much weights should be changed when
                    training network. Google it, i am not explaining any further (number)
    hidden_layers_count - Hidden layers count (int)
    hidden_layers_width - Hidden layers width (int)
    activation - Activation function (def)
    activation_derivative - Activation derivative function (def)


  train(self, inputs, targets, *_, epochs=20000) -> accuracy
    trains neural network. i am not explaining how, just
    google it. this is documentation, not guide or paper

    inputs - Training inputs, that are fed
             to neural network (list of NeuralInput)
    targets - Training outputs, that are
              targets for neural network to hit (list of NeuralInput)
    epochs - Number of epochs to run training.
             More epochs = more accurate, but still can
             ruin entire network if too much. (int)
    accuracy - Network accuracy, based on last epoch error. (number)


  predict(self, input_data) -> output
    predict the answer based on input

    input_data - neural network input data (list of NeuralInput)
    output - prediction (list of numbers)


  save(self, path)
    saves neural network to file

    path - path to where to save (string)



