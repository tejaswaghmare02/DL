# %%
# Implement feed forward single layer perceptron with suitable example.

# %%
# single layer perceptron for OR gate
import numpy as np

def activate(x):
    return 1 if x >0 else 0

# Define the single-layer perceptron function
def single_layer_perceptron(inputs, weights):
    weighted_sum = np.dot(inputs, weights)
    output = activate(weighted_sum)
    return output

# Define the logical OR training data
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_outputs = np.array([0, 1, 1, 1])

weights=np.random.randn(2)

learning_rate = 0.1
epochs = 100

for epoch in range(epochs):
    for inputs, expected_output in zip(training_inputs, expected_outputs):
        prediction = single_layer_perceptron(inputs, weights)
        error = expected_output - prediction
        
        # Update weights based on the error
        weights += learning_rate * error * inputs

print("Trained weights:", weights)
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
print("\nTesting the trained perceptron:")
for inputs in test_inputs:
    output = single_layer_perceptron(inputs, weights)
    print(f"Input: {inputs} Output: {output}")
