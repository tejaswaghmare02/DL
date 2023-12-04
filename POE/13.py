# %%
"""
# MNIST Digit Classification with Data Shuffling
"""

# %%
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


# %%

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() 

train_images, test_images = train_images / 255.0, test_images / 255.0

# Shuffle the training data
shuffled_indices = np.random.permutation(len(train_images))
train_images, train_labels = train_images[shuffled_indices], train_labels[shuffled_indices]


# %%

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# %%

# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))


# %%

# Generate and display training curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
