# %%
"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): Wangshu Cai (u7546753)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    # print(f'conv filter size: {conv_filter.shape}')
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros(img.shape)
    img = np.pad(img, ((pad, pad), (pad, pad)), 'edge')
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


def HarrisCornerness(Ix2, Ixy, Iy2, k=0.04):
    # Initialize a zero matrix with shape [image shape]x2x2 to store M.
    M = np.zeros((Ix2.shape[0], Ix2.shape[1], 2, 2))
    # Initialize cornerness matrix R with shape [image shape]
    R = np.zeros(Ix2.shape)
    # Iterate through the image (row)
    for i in range(Ix2.shape[0]):
        # Iterate through the image (column)
        for j in range(Ix2.shape[1]):
            # Update values of M for each pixel equals to w[[Ix^2,IxIy],[Ix,Iy,Iy^2]]
            M[i, j, :, :] = np.array([[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]])
            # Update R value with M equals to det(M) - k * trace(M)^2
            R[i, j] = np.linalg.det(M[i, j]) - k * (np.trace(M[i, j])) ** 2
    # return cornerness matrix
    return R


def non_max_suppression(R, border_width=5, threshold=0.01):
    # pad R with border width, replicate edge values to border
    R_border = cv2.copyMakeBorder(R, border_width, border_width, border_width, border_width, cv2.BORDER_REPLICATE)
    # corner values after suppression
    R_star = R_border[:, :]
    # Iterate through the matrix (row)
    for i in range(border_width, R_border.shape[0] - border_width):
        # Iterate through the matrix (column)
        for j in range(border_width, R_border.shape[1] - border_width):
            # if the center point is not the maximum in the square matrix with 2*width + 1 length around it
            if np.max(R_border[i - border_width:i + border_width + 1, j - border_width:j + border_width + 1]) != \
                    R_border[
                        i, j]:
                # supress the point
                R_star[i, j] = 0
    # Adjust the threshold according to the values in R_star
    Threshold = np.max(R_star) * threshold
    # Initialize a boolean matrix with False (0)
    Final_mark = np.zeros(R.shape)
    # Set the value to True (1) if the suppressed cornerness value is greater than threshold
    Final_mark += R_star[border_width:-border_width, border_width:-border_width] > Threshold
    # Set the datatype to Boolean
    Final_mark = Final_mark.astype(np.dtype('bool'))
    # Index of corners
    index = np.argwhere(Final_mark > 0)
    return index

#%%
# Parameters
sigma = 2
thresh = 0.01
k = 0.04
image_dir = 'Harris-6.jpg'
circle_size = 2
# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()



# Returns two arrays containing the index of dectected corners of my function and inbuilt function
def compute_harris(image_dir, sigma, thresh, k):
    # Read from image
    bw = cv2.imread(image_dir)
    # Change the color scheme to grayscale
    bw = cv2.cvtColor(bw, cv2.COLOR_BGR2GRAY)
    # Change the range of the image from float in [0, 1] to integer in [0, 255]
    bw = np.array(bw * 255, dtype=int)
    # Compute x and y derivatives of image
    Ix = conv2(bw, dx)
    Iy = conv2(bw, dy)

    # Obtain gaussian window with size of 3 sigma + 1 (covers 99.7% of the bell shaped distribution)
    g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)

    # Perform convolution on Ix, Iy to get wIx^2, wIy^2, wIxIy
    Iy2 = conv2(np.power(Iy, 2), g)
    Ix2 = conv2(np.power(Ix, 2), g)
    Ixy = conv2(Ix * Iy, g)
    # image read for inbuilt function
    img_inbuilt = np.float32(cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2GRAY))
    # compute Harris cornerness
    R = HarrisCornerness(Ix2, Ixy, Iy2)
    R_inbuilt = cv2.cornerHarris(img_inbuilt, 13, 3, k)
    final_index = non_max_suppression(R,3,threshold=thresh)
    final_index_inbuilt = non_max_suppression(R_inbuilt,3,threshold=thresh)
    return final_index, final_index_inbuilt



Final_index, Final_index_inbuilt = compute_harris(image_dir,sigma,thresh,k)
Final_image = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)

Final_image_inbuilt = cv2.cvtColor(cv2.imread(image_dir), cv2.COLOR_BGR2RGB)
# Draw red circles to indicate corners
for i in Final_index:
    cv2.circle(Final_image, (i[1], i[0]), circle_size, (255, 0, 0), -1)
for i in Final_index_inbuilt:
    cv2.circle(Final_image_inbuilt, (i[1], i[0]), circle_size, (255, 0, 0), -1)
# show image with corners
plt.imshow(Final_image)
plt.title(f'Harris Corners for {image_dir}')
plt.show()
plt.imshow(Final_image_inbuilt)
plt.title(f'Inbuilt Harris Corners for {image_dir}')
plt.show()
