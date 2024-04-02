class Activations:
    def __call__(self, x):
        # Call the activation function
        raise NotImplementedError("Subclass implementation")

    def derivative(self, x):
        raise NotImplementedError("Subclass implementation")

class Sigmoid(Activations):
    def __call__(self, x):
        # Sigmoid activation function
        pass

    def derivative(self, x):
        # Derivative of sigmoid activation function
        pass

class ReLU(Activations):
    def __call__(self, x):
        # ReLU activation function
        pass

    def derivative(self, x):
        # Derivative of ReLU activation function
        pass

class tanH(Activations):
    def __call__(self, x):
        # tanH activation function
        pass

    def derivative(self, x):
        # Derivative of tanH activation function
        pass