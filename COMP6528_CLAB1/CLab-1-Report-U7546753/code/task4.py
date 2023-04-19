# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
# read the original image
img3 = cv2.imread('image3.jpg', cv2.IMREAD_UNCHANGED)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
plt.imshow(img3, cmap='gray')
plt.title('original image')
plt.show()


# %%
# sobel filter
def sobel_filter(img):
    # create the sobel kernel with size 3x3
    sobel_kernel_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # padding the image
    img_padding = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    img_padding[1:-1, 1:-1] += img
    # apply the sobel filter in two direction and merge the result
    ans = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img_padding.shape[0] - 2):
        for j in range(img_padding.shape[1] - 2):
            a = np.sum(img_padding[i:i + 3, j:j + 3] * sobel_kernel_horizontal)
            b = np.sum(img_padding[i:i + 3, j:j + 3] * sobel_kernel_vertical)
            ans[i,j] = np.sqrt(a ** 2 + b ** 2)
    # normalize the result
    ans = np.clip(ans,0,255).astype(np.uint8)
    return ans


img3_sobel = sobel_filter(img3)
img3_sobel[img3_sobel > 70] = 255
img3_sobel[img3_sobel <= 70] = 0
plt.imshow(img3_sobel, cmap='gray')
plt.title('sobel filter')
plt.show()

# %% inbuilt sobel
img3_sobel_inbuilt = cv2.Sobel(img3, -1, 0, 1, ksize=3)
img3_sobel_inbuilt1 = cv2.Sobel(img3, -1, 1, 0, ksize=3)
threshold = 25
img3_sobel_inbuilt[img3_sobel_inbuilt > threshold] = 255
img3_sobel_inbuilt[img3_sobel_inbuilt <= threshold] = 0
img3_sobel_inbuilt1[img3_sobel_inbuilt > threshold] = 255
img3_sobel_inbuilt1[img3_sobel_inbuilt <= threshold] = 0
img3_sobel_inbuilt = np.clip(np.sqrt(img3_sobel_inbuilt1 ** 2 + img3_sobel_inbuilt ** 2),0,255).astype(np.uint8)
# normalize the result
plt.imshow(img3_sobel_inbuilt, cmap='gray')
plt.title('inbuilt sobel filter')
plt.show()

#%% orientation
# horizontal and vertical sobel filter
img3_sobel_inbuilt_h = cv2.Sobel(img3, -1, 0, 1, ksize=3)
img3_sobel_inbuilt_v = cv2.Sobel(img3, -1, 1, 0, ksize=3)
# calculate the orientation
orientation = (np.arctan2(img3_sobel_inbuilt_v, img3_sobel_inbuilt_h)/np.pi*180).astype(np.uint8)
# construct the histogram
hist = np.zeros(91)
for i in range(img3_sobel_inbuilt_v.shape[0]):
    for j in range(img3_sobel_inbuilt_v.shape[1]):
        if img3_sobel_inbuilt_v[i,j] != 0 or img3_sobel_inbuilt_h[i,j] != 0:
            hist[orientation[i,j]] += 1
# plot the histogram
plt.plot(hist)
plt.title('orientation histogram')
plt.xlim(-10,100)
plt.show()