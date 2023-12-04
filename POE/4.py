# %%
"""
#  Implement McCulloch Pitts neural network using Tensorflow (Explain with suitable example )
"""

# %%
import numpy as np

def mcculloch_pitts_neuron(input_data, weights, threshold):
    weighted_sum = np.dot(input_data, weights)
    output = np.where(weighted_sum >= threshold, 1, 0)
    return output

# Define the input data (truth table for AND gate)
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the weights
weights = np.array([1, 1])

output = mcculloch_pitts_neuron(input_data, weights, 2)
print("Input Data:\n", input_data)
print("AND Gate Output:\n", output)
