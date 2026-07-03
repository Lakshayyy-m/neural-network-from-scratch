import numpy as np


class BaseLayer:
    """Base class for all layers providing a common interface."""

    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dvalues):
        raise NotImplementedError


def calculate_accuracy(output, y):
    """Calculate accuracy from model output and true labels.

    Handles both sparse labels and one-hot encoded targets.

    Parameters
    ----------
    output : np.ndarray
        Model output (probabilities or logits), shape (samples, classes).
    y : np.ndarray
        True labels, either sparse (shape (samples,)) or one-hot encoded
        (shape (samples, classes)).

    Returns
    -------
    float
        Accuracy as a value between 0 and 1.
    """
    predictions = np.argmax(output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    return accuracy
