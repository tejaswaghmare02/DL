# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image using cv2.imread
image_path = "flower.jpg"
image_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 

# Check if the image is loaded successfully
if image_cv2 is None:
    print("Error: Unable to load the image.")
    exit()

# Vertical edge detection filter (Sobel filter)
vertical_edge_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]], dtype=np.float32)

# Horizontal edge detection filter (Sobel filter)
horizontal_edge_filter = np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]], dtype=np.float32)

# Apply vertical edge detection filter using filter2D
vertical_edges = cv2.filter2D(image_cv2, -1, vertical_edge_filter)

# Apply horizontal edge detection filter using filter2D
horizontal_edges = cv2.filter2D(image_cv2, -1, horizontal_edge_filter)

# Display original and edge-detected images
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image_cv2, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(vertical_edges, cmap='gray')
plt.title('Vertical Edges')

plt.subplot(1, 3, 3)
plt.imshow(horizontal_edges, cmap='gray')
plt.title('Horizontal Edges')

plt.tight_layout()
plt.show()
