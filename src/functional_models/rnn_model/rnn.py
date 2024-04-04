import activations
import tensorflow as tf

class RNNLM:
    """
    Recurrent Neural Network (RNN) Language Model.

    This class represents a language model implemented using a recurrent neural network (RNN).

    Parameters:
    ----------
    num_inputs : int
        The dimensionality of the input feature space, typically the size of the vocabulary.

    num_hiddens : int
        The number of hidden units in each recurrent layer of the RNN. (width)

    num_layers : int
        The number of recurrent layers stacked on top of each other. (depth)

    vocab_size : int
        The size of the vocabulary, i.e., the number of unique tokens in the language.

    activation_function : Activations
        An instance of the `Activations` class specifying the activation function to be used in the RNN.

    sigma : float, optional (default=0.01)
        The standard deviation of the normal distribution used for weight initialization. not sure about other ways to initialize

    Attributes:
    -----------
    num_inputs : int
        The dimensionality of the input feature space.

    num_hiddens : int
        The number of hidden units in each recurrent layer.

    num_layers : int
        The number of recurrent layers.

    vocab_size : int
        The size of the vocabulary.

    activation_function : Activations
        An instance of the `Activations` class specifying the activation function used in the RNN.

    sigma : float
        The standard deviation of the normal distribution used for weight initialization.

    Methods:
    --------
    none yet

    """

    def __init__(self, num_inputs, num_hiddens, num_layers, vocab_size, activation_function, sigma=0.01):
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.activation_function = activation_function
        self.sigma = sigma
        self.initialize_params()

    def initialize_params(self):
        self.W_xh = tf.Variable(tf.random.normal(
            (self.num_inputs, self.num_hiddens)) * self.sigma) # Initialize input-to-output weights
        self.W_hh = tf.Variable(tf.random.normal(
            (self.num_hiddens, self.num_hiddens)) * self.sigma) # Initialize hidden-to-hidden weights
        self.W_hq = tf.Variable(tf.random.normal(
            (self.num_hiddens, self.vocab_size)) * self.sigma)  # Initialize hidden-to-output weights
        self.b_h = tf.Variable(tf.zeros(self.num_hiddens))   # Initialize hidden biases
        self.b_q = tf.Variable(tf.zeros(self.vocab_size))   # Initialize output biases


    def forward(self, x):
        output, state = 0, 0
        # output to output and state is the hidden state
        return output, state

    def train():
        pass
