# %%
"""
# Using a Pre-trained ImageNet Network
"""

# %%
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# Load a pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Define your custom image preprocessing function (for your specific dataset)
def preprocess_custom_image(image_path):
    # Load and preprocess your custom image
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Load and preprocess a custom image for classification
custom_image_path = 'flower.jpg'  # Replace with your image path
custom_image = preprocess_custom_image(custom_image_path)

# Make predictions using the pre-trained model
predictions = model.predict(custom_image)

# Decode the predictions to human-readable labels
decoded_predictions = decode_predictions(predictions, top=5)[0]

# Print the top predicted labels and their associated probabilities
for label, description, probability in decoded_predictions:
    print(f"{label}: {description} ({probability:.2f})")


# %%
