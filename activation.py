import numpy as np


class Activation_ReLU:
    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"'inputs' must be a numpy array, got {type(inputs).__name__}")
        if inputs.size == 0:
            raise ValueError("'inputs' must not be empty")
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        if not isinstance(inputs, np.ndarray):
            raise TypeError(f"'inputs' must be a numpy array, got {type(inputs).__name__}")
        if inputs.size == 0:
            raise ValueError("'inputs' must not be empty")
        if inputs.ndim < 2:
            raise ValueError(
                f"'inputs' must be at least 2D (batch of samples), got {inputs.ndim}D"
            )
        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # this ensures that we keeping the dimensions the same, meaning that the max is not calculated overall the batch outputs but per outputs. And then ofcourse the axis is related to which axis to calcualte the max in(here we are taking it as row)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        if not isinstance(dvalues, np.ndarray):
            raise TypeError(f"'dvalues' must be a numpy array, got {type(dvalues).__name__}")
        if not hasattr(self, 'output'):
            raise RuntimeError(
                "backward() called before forward() — no output to compute gradients from"
            )

        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            #Flatten output array
            single_output = single_output.reshape(-1, 1)

            #calculate Jacobian matrix of the output and calculate sample wise gradient and add it to the array of sample gradients
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
## By now I have understood calculating the derivative of the Common Categorical Loss function with chain rule, that is further calculating the Partial derivative of the Softmax Activation function as well. Next step would be the code implementation of this. I suggest reading the last defined equation once again in the PDF. Page number I am on in PDF -> 230 (as per computer). Topic -> Common Categorical Cross-Entropy loss and Softmax activation derivative - code implementation
