import numpy as np


class Loss:
    def calculate(self, output, y):
        if not isinstance(output, np.ndarray):
            raise TypeError(f"'output' must be a numpy array, got {type(output).__name__}")
        if not isinstance(y, np.ndarray):
            raise TypeError(f"'y' must be a numpy array, got {type(y).__name__}")
        if output.size == 0:
            raise ValueError("'output' must not be empty")
        if y.size == 0:
            raise ValueError("'y' must not be empty")
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        if samples != len(y_true):
            raise ValueError(
                f"Number of predictions ({samples}) does not match "
                f"number of targets ({len(y_true)})"
            )
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # need to check if the categories passed are scalar or a hot one encoded vector
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        else:
            raise ValueError(
                f"'y_true' must be 1D (sparse labels) or 2D (one-hot encoded), "
                f"got {len(y_true.shape)}D"
            )

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood

    # backward pass
    def backward(self, dvalues, y_true):
        if not isinstance(dvalues, np.ndarray):
            raise TypeError(f"'dvalues' must be a numpy array, got {type(dvalues).__name__}")
        if not isinstance(y_true, np.ndarray):
            raise TypeError(f"'y_true' must be a numpy array, got {type(y_true).__name__}")
        if dvalues.size == 0:
            raise ValueError("'dvalues' must not be empty")
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        if np.any(dvalues == 0):
            raise ValueError(
                "'dvalues' contains zeros, which would cause division by zero "
                "in gradient calculation"
            )
        self.dinputs = -y_true / dvalues
        # Normalise gradient
        self.dinputs = self.dinputs / samples

