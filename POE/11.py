# %%
#Implement CNN operation to horizontal edge detection of an Image.

# %%
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

# %%
image_path = "flower.jpg" 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

image_resized = cv2.resize(image, (640, 640))
plt.imshow(image_resized,cmap='gray')

# %%

# Reshape the image to match the expected input shape for convolution
image = image_resized.reshape(1, 640, 640, 1)

# %%
filter=np.array([[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]], dtype=np.float32)

# %%
# Create TensorFlow constants for the image and the filters
image_tensor = tf.constant(image,dtype=np.float32)
filter_tensor = tf.constant(filter.reshape(3, 3, 1, 1), dtype=tf.float32)

convolution = tf.nn.conv2d(image_tensor, filter_tensor, strides=1, padding='SAME')
convolution_output =convolution.numpy().squeeze()

plt.imshow(convolution_output,cmap='gray')


# %%
