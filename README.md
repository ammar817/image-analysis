# IMAGE_ANALYSIS
# importing libraries
import cv2
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('/content/Beast-Boy-PNG-Free-Download.png')

# Convert the image to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply a Gaussian blur filter
blurred_image = cv2.GaussianBlur(image_rgb, (15, 15), 0)

# Apply a Canny edge detector
edges = cv2.Canny(image_rgb, 100, 200)

# Convert the edge-detected image to RGB for displaying with matplotlib
edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

# Display the original and filtered images using matplotlib
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

# Blurred Image
plt.subplot(1, 3, 2)
plt.title('Blurred Image')
plt.imshow(blurred_image)
plt.axis('off')

# Edge Detected Image
plt.subplot(1, 3, 3)
plt.title('Edge Detected Image')
plt.imshow(edges_rgb)
plt.axis('off')
# "HISTOGRAM"
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('/content/Beast-Boy-PNG-Free-Download.png')

# Convert the image to RGB (OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate the histogram for each color channel
colors = ('r', 'g', 'b')
channels = cv2.split(image_rgb)

# Plot the histograms
plt.figure(figsize=(15, 5))
plt.title('Color Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

for (channel, color) in zip(channels, colors):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(histogram, color=color)

# Display the histogram
plt.show()

# "ROTATED IMAGE"
import cv2
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('/content/Beast-Boy-PNG-Free-Download.png')

# Get the dimensions of the image
(h, w) = image.shape[:2]

# Calculate the center of the image
center = (w // 2, h // 2)

# Rotate the image by 45 degrees
M = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h))

# Convert the image to RGB for displaying
rotated_image_rgb = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)

# Display the original and rotated images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Rotated Image')
plt.imshow(rotated_image_rgb)
plt.axis('off')

plt.show()
