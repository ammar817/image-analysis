# IMAGE_ANALYSIS
# importing libraries
import cv2 as cv
from matplotlib import pyplot as plt

image = cv.imread('/content/Beast-Boy-PNG-Free-Download.png')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
blured_image = cv.GaussianBlur(image_rgb, (5, 5), 0)
edge_rgb = cv.Canny(blured_image, 100, 200)
plt.imshow(edge_rgb)
plt.show()
# size of picture
image.size
# shape of picture
image.shape
