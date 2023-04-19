# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Read images
img1 = cv2.imread('image1.jpg', cv2.IMREAD_UNCHANGED)
img2 = cv2.imread('image2.jpeg', cv2.IMREAD_UNCHANGED)
img3 = cv2.imread('image3.jpg', cv2.IMREAD_UNCHANGED)

# %% a
# resize the image into 384x256
img1_resized = cv2.resize(img1, (384, 256))
# Show image from bgr to rgb
plt.imshow(img1_resized[:,:,::-1])
plt.show()

# %% b
# show the blue channel
plt.imshow(img1_resized[:, :, 0],cmap='gray')
plt.title('grayscale image of B channel')
plt.show()
# show the green channel
plt.imshow(img1_resized[:, :, 1],cmap='gray')
plt.title('grayscale image of G channel')
plt.show()
# show the red channel
plt.imshow(img1_resized[:, :, 2],cmap='gray')
plt.title('grayscale image of R channel')
plt.show()

# %% c
color = ('b', 'g', 'r')
# loop for each color
for i, col in enumerate(color):
    # Generate histogram with specific channel.
    hist = cv2.calcHist([img1_resized], [i], None, [256], [0, 256])
    # Plot the histogram
    plt.plot(hist, color=col)
    plt.xlim([0, 256])
    plt.title(col + ' histogram')
    plt.show()
# %% d
# Convert the image to grayscale
img1_gray = cv2.cvtColor(img1_resized, cv2.COLOR_RGB2GRAY)
img1_gray_eq = cv2.equalizeHist(img1_gray)
plt.imshow(img1_gray_eq, cmap='gray')
plt.title('hist equalized grayscale image')
plt.show()
# Plot the histogram
plt.plot(cv2.calcHist([img1_gray_eq], [0], None, [256], [0, 256]), color='gray')
plt.xlim([0, 256])
plt.title('gray histogram')
plt.show()

# Convert the image to green channel
img1_g_eq = cv2.equalizeHist(img1_resized[:, :, 0])
plt.imshow(img1_g_eq, cmap='gray')
plt.title('hist equalized green image')
plt.show()
# Plot the histogram
plt.title('hist equalized green image')
plt.plot(cv2.calcHist([img1_g_eq], [0], None, [256], [0, 256]), color='g')
plt.xlim([0, 256])
plt.title('green histogram')
plt.show()

# Convert the image to blue channel
img1_b_eq = cv2.equalizeHist(img1_resized[:, :, 1])
plt.imshow(img1_b_eq, cmap='gray')
plt.title('hist equalized blue image')
plt.show()
plt.plot(cv2.calcHist([img1_b_eq], [0], None, [256], [0, 256]), color='b')
plt.xlim([0, 256])
plt.title('blue histogram')
plt.show()

img1_r_eq = cv2.equalizeHist(img1_resized[:, :, 2])
plt.imshow(img1_r_eq, cmap='gray')
plt.title('hist equalized red image')
plt.show()
plt.plot(cv2.calcHist([img1_r_eq], [0], None, [256], [0, 256]), color='r')
plt.xlim([0, 256])
plt.title('red histogram')
plt.show()
