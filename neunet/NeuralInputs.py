"""
Base neural network input class. This is standartised
so for image you may want to use Neural2DInput
"""
class NeuralInput:
    def __init__(self, data):
        self.data = data

    def get(self):
        return self.data

"""
2d neural input.
"""
class Neural2DInput(NeuralInput):
    def __init__(self, pixels):
        self.data = [pixels, "raw"]

    def flatten(self):
        flat = [pix for line in self.data[0] for pix in line]
        self.data = [flat, "flat"]

    def get(self):
        return self.data[0]

    def normalise(self):
        if self.data[1] != "flat": raise NeuralError("2DInput should be .flatten() before normalising!")
        for i in range(0, len(self.data[0])):
            pixel = self.data[0][i]
            self.data[0][i] = (sum(pixel) / len(pixel)) / max(self.data[0])
        self.data[1] = "norm"
