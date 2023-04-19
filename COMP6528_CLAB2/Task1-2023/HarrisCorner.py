# %%
"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID):
"""

import numpy as np
import cv2


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    # print(f'conv filter size: {conv_filter.shape}')
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01

# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

#%%
bw = cv2.imread('Harris-1.jpg')
bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
bw = np.array(bw * 255, dtype=int)
# computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)


# %%
######################################################################
# Task: Compute the Harris Cornerness
######################################################################
def HarrisCornerness(Ix2, Ixy, Iy2, k=0.04):
    M = np.zeros((Ix2.shape[0], Ix2.shape[1], 2, 2))
    R = np.zeros(Ix2.shape)
    for i in range(Ix2.shape[0]):
        for j in range(Ix2.shape[1]):
            M[i, j, :, :] = np.array([[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]])
            R[i, j] = np.linalg.det(M[i, j]) - k * (np.trace(M[i, j])) ** 2
    return R



######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################

R = HarrisCornerness(Ix2, Ixy, Iy2)

R_border = cv2.copyMakeBorder(R, 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=0)
R_star = R_border[:,:]
for i in range(1,R_border.shape[0]-1):
    for j in range(1,R_border.shape[1]-1):
        if np.max(R_border[i-1:i+2,j-1:j+2])!=R_border[i,j]:
            R_star[i,j]=0
Threshold = np.max(R_star) * thresh
Final_mark = R_star[1:-1,1:-1] > Threshold
corner_count = np.count_nonzero(Final_mark)
Final_image = cv2.cvtColor(cv2.imread('Harris-1.jpg'),cv2.COLOR_BGR2RGB)
Final_image[Final_mark,:] = np.array([255,0,0])

plt.imshow(Final_image)
plt.show()