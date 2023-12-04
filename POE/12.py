# %%
# Implement A simple CNN
# Make a train and validation dataset of images with vertical and horizontal images 
# Defining the CNN to predict the knowledge from image classification Visualising the learned CNN Model.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic images of vertical and horizontal lines
def generate_images(size=100, line_width=3):
    vertical_lines = np.zeros((size, size))
    horizontal_lines = np.zeros((size, size))

    vertical_lines[:, size // 2 - line_width // 2:size // 2 + line_width // 2] = 1
    horizontal_lines[size // 2 - line_width // 2:size // 2 + line_width // 2, :] = 1

    return vertical_lines, horizontal_lines

# Generate synthetic images
num_images = 1000  # Number of images for each class
vertical_images = [generate_images()[0] for _ in range(num_images)]
horizontal_images = [generate_images()[1] for _ in range(num_images)]

# Display sample images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(vertical_images[0], cmap='gray')
plt.title('Vertical Line Image')

plt.subplot(1, 2, 2)
plt.imshow(horizontal_images[0], cmap='gray')
plt.title('Horizontal Line Image')

plt.tight_layout()
plt.show()


# %%
from sklearn.model_selection import train_test_split

# Combine images and labels
images = np.array(vertical_images + horizontal_images)
labels = np.array([0] * num_images + [1] * num_images)  # 0 for vertical, 1 for horizontal

# Split dataset into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Reshape the images for compatibility with CNN
train_images = train_images.reshape(-1, 100, 100, 1)
val_images = val_images.reshape(-1, 100, 100, 1)


# %%
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))


# %%
# Visualize the learned filters in the first convolutional layer
filters, biases = model.layers[0].get_weights()
plt.figure(figsize=(8, 8))
for i in range(filters.shape[3]):
    plt.subplot(6, 6, i + 1)
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()
