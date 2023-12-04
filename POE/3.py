# %%
# Implement Linear Algebra Tensors Tensor arithmetic Implementing matrix multiplication (minimum four possible operations are necessary) .

# %%
# Using Numpy

import numpy as np

# Create tensors (multi-dimensional arrays) using NumPy
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print('tensor_a\n',a)
print('tensor_b\n',b)

# %%
# Addition
tensor_sum = a + b
print("Sum of tensors:\n",tensor_sum)

# Scalar multiplication
scalar = 2
tensor_scaled = scalar * a
print("\nScalar multiplication:\n",tensor_scaled)

# Element-wise multiplication
tensor_product = np.multiply(a,b) #a*b
print("\nElement-wise multiplication:\n",tensor_product)

#matrix multiplication
dot_product=a@b # equivalent to np.dot(a,b)
print("\nMatrix multiplication:\n",dot_product)

# %%

# Using Tensorflow

import tensorflow as tf

# Create tensors using TensorFlow
tensor_a = tf.constant([[1, 2], [3, 4]])
tensor_b = tf.constant([[5, 6], [7, 8]])

# Addition
tensor_sum = tf.math.add(tensor_a, tensor_b)
print("Sum of tensors:\n",tensor_sum)

# Scalar multiplication
scalar = 2
tensor_scaled = tf.math.scalar_mul(scalar, tensor_a)
print("Scalar multiplication:\n",tensor_scaled)

# Element-wise multiplication
tensor_product = tf.math.multiply(tensor_a, tensor_b)
print("\nElement-wise multiplication:\n",tensor_product)

#dot product
dot_product = tf.linalg.matmul(tensor_a, tensor_b)
print("\nDot Product:\n",dot_product)

