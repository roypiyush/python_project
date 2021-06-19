# Python program to implement a
# single neuron neural network
from numpy import array, random, dot, tanh


class Neuron(object):
    """
    Represents single neuron which accepts 3 array_of_inputs to decide 1 output. Neuron needs to be trained first in
    order to produce expected result. Training involves calculating the error then back propagating the gradients. The
    step of calculating the output of neuron is called forward propagation while calculation of gradients is called
    back propagation.
    """

    def __init__(self, number_of_input_weights, number_of_output_weights):
        # 3x1 Weight matrix. Weights for input vector. Mathematically, this is linear equation of n=3 variables
        self.weight_matrix = 2 * random.random((number_of_input_weights, number_of_output_weights)) - 1

    @staticmethod
    def activation_function(x):
        """
        Hyperbolic tangent as activation function
        :param x:
        :return:
        """
        return tanh(x)

    @staticmethod
    def gradient_descent(x):
        """
        derivative of activation function.
        Needed to calculate the gradients.
        :param x:
        :return:
        """
        return 1.0 - tanh(x) ** 2

    def forward_propagation(self, array_of_inputs):
        """
        1. Multiplying each of inputs with corresponding weights
        2. Use activation function to calculate final output of neuron.
        :param array_of_inputs:
        :return:
        """
        # Below is transfer function. Sum of weighted array_of_inputs.
        # Solving equation a1.x^n + a2.x^n-1 + ... + an.x + a0
        dot_product = dot(array_of_inputs, self.weight_matrix)
        return self.activation_function(dot_product)   # Transfer function is passed to activation function

    def train(self, train_inputs, train_outputs,
              num_train_iterations):
        """
        Training involves two steps.
        1. Do forward propagation
        2. Do Back propagation
        :param train_inputs:
        :param train_outputs:
        :param num_train_iterations:
        :return:
        """

        # Number of iterations we want to
        # perform for this set of input.
        for iteration in range(num_train_iterations):
            self._train(train_inputs, train_outputs)

    def _train(self, train_inputs, train_outputs):
        outputs = self.forward_propagation(train_inputs)
        # Now do back propagation.
        # 1. Calculate error. Deviation from expected output
        # 2. Calculate gradient descent of outputs of each output.
        # 3. Multiply gradient with error for each input. This is done because we want to adjust more if error is
        #    high or less is error is less.
        # 4. Find adjustments for weights against each input
        # 5. Perform the adjustment of weights.
        errors = train_outputs - outputs
        adjustments = dot(train_inputs.T, errors * self.gradient_descent(outputs))
        # Adjust each of coefficients to linear equation
        self.weight_matrix += adjustments


def main():
    single_neuron = Neuron(number_of_input_weights=3, number_of_output_weights=1)
    # 4 Different input is provided. Each Input contains 3 axons
    train_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    train_outputs = array([[0, 1, 1, 0]]).T
    print('Initial Weights {}'.format(single_neuron.weight_matrix.T))

    single_neuron.train(train_inputs, train_outputs, 10000)
    print('Final Weights {}'.format(single_neuron.weight_matrix.T))

    # Test the neural network with a new situation.
    new_input0 = array([0, 0, 0])
    result = single_neuron.forward_propagation(new_input0)
    print("New input0 after training {}, Result {}".format(new_input0, result))

    new_input1 = array([0, 0, 1])
    result = single_neuron.forward_propagation(new_input1)
    print("New input1 after training {}, Result {}".format(new_input1, result))

    new_input2 = array([0, 1, 0])
    result = single_neuron.forward_propagation(new_input2)
    print("New input2 after training {}, Result {}".format(new_input2, result))

    new_input3 = array([0, 1, 1])
    result = single_neuron.forward_propagation(new_input3)
    print("New input3 after training {}, Result {}".format(new_input3, result))

    new_input4 = array([1, 0, 0])
    result = single_neuron.forward_propagation(new_input4)
    print("New input4 after training {}, Result {}".format(new_input4, result))

    new_input5 = array([1, 0, 1])
    result = single_neuron.forward_propagation(new_input5)
    print("New input5 after training {}, Result {}".format(new_input5, result))

    new_input6 = array([1, 1, 0])
    result = single_neuron.forward_propagation(new_input6)
    print("New input6 after training {}, Result {}".format(new_input6, result))

    new_input7 = array([1, 1, 1])
    result = single_neuron.forward_propagation(new_input7)
    print("New input7 after training {}, Result {}".format(new_input7, result))


if __name__ == "__main__":
    main()
