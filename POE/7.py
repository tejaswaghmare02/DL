# %%
# Implement multilayer perceptron with back propagation (student should implement mathematical operations of back propagation with suitable example).

# %%
"""
### Mathematical Operations in Backpropagation:

#### Forward Propagation:
1. **Weighted Sum at Each Neuron:**
   \[ z = \sum_{i=1}^{n} (w_i * x_i) + b \]
   \(w_i\) = weights, \(x_i\) = inputs, \(b\) = bias

2. **Activation Function:**
   \[ a = \sigma(z) \]
   \(a\) = output of the neuron, \(\sigma\) = activation function

#### Backpropagation:
1. **Output Layer Error:**
   \[ \delta_{output} = (y_{true} - y_{predicted}) \times \sigma'(z_{output}) \]
   
2. **Hidden Layer Error:**
   \[ \delta_{hidden} = \delta_{output} \times W_{output}^T \times \sigma'(z_{hidden}) \]

3. **Weight and Bias Update:**
   \[ \Delta W = \alpha \times \delta \times X \]

"""

# %%
import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Set the input data (XOR)
X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

# Set the true outputs sum of two numbers
y_true = np.array([[0],[0],[0],[1]])

# Initialize weights and biases randomly
input_size = 2
hidden_size = 4
output_size = 1

hidden_weights = np.random.uniform(size=(input_size, hidden_size))
hidden_bias = np.random.uniform(size=(1, hidden_size))
output_weights = np.random.uniform(size=(hidden_size, output_size))
output_bias = np.random.uniform(size=(1, output_size))

learning_rate = 0.1
epochs = 10000

# Training the network
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    # Backpropagation
    output_error = y_true - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)

    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    # Updating weights and biases
    output_weights += learning_rate * hidden_layer_output.T.dot(output_delta)
    output_bias += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
    hidden_weights += learning_rate * X.T.dot(hidden_delta)
    hidden_bias += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

# Testing the trained network
hidden_layer = sigmoid(np.dot(X, hidden_weights) + hidden_bias)
output_layer = sigmoid(np.dot(hidden_layer, output_weights) + output_bias)

print("Predicted Output after Training:")
print(output_layer)
threshold = 0.5
rounded_output = np.round(output_layer)
rounded_output = (output_layer > threshold).astype(int)

print("Rounded Output:")
print(rounded_output)

