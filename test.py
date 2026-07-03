import numpy as np
from utils import calculate_accuracy

# Probabilities of 3 samples
softmax_outputs = np.array([[0.7, 0.2, 0.1],
                            [0.5, 0.1, 0.4],
                            [0.02, 0.9, 0.08]])
# Target (ground-truth) labels for 3 samples
class_targets = np.array([0, 1, 1])

accuracy = calculate_accuracy(softmax_outputs, class_targets)

predictions = np.argmax(softmax_outputs, axis=1)
print(predictions)
print(predictions == class_targets)
print('acc:', accuracy)
