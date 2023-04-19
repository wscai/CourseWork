# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2

img2 = cv2.imread('image2.jpeg', cv2.IMREAD_UNCHANGED)
plt.imshow(img2[:, :, ::-1])
plt.title('original image')
plt.show()
# %% 1
# calculate the cropping start point
cropping_start = (img2.shape[1] - img2.shape[0]) // 2
# crop the image
img2_crop = img2[:, cropping_start:cropping_start + img2.shape[0], :]
# resize the image
img2_crop = cv2.resize(img2_crop, (512, 512))
# convert to gray scale
img2_crop = cv2.cvtColor(img2_crop, cv2.COLOR_BGR2GRAY)
# show the image
plt.imshow(img2_crop, cmap='gray')
plt.title('cropped and gray scaled image')
plt.show()

# %% 2
mu = 0
sigma = 15
# Generate gaussian noise with mu and sigma
gaussian_noise = mu + np.random.randn(img2_crop.shape[0], img2_crop.shape[1]) * sigma
# Add noise to the image
img2_with_noise = img2_crop + gaussian_noise
# Clib the values into [0, 255]
img2_with_noise[img2_with_noise > 255] = 255
img2_with_noise[img2_with_noise < 0] = 0
# Convert to correct data type
img2_with_noise = img2_with_noise.astype(np.uint8)
plt.imshow(img2_with_noise, cmap='gray')
plt.title('image with gaussian noise')
plt.show()

# %% 3
# Generate histogram of the original image
img2_hist = cv2.calcHist([img2_crop], [0], None, [256], [0, 256])
plt.plot(img2_hist, color='gray')
plt.xlim([0, 256])
plt.title('img2_crop histogram')
plt.show()
# Generate histogram of the image with noise
img2_with_noise_hist = cv2.calcHist([img2_with_noise], [0], None, [256], [0, 256])
plt.plot(img2_with_noise_hist, color='red')
plt.xlim([0, 256])
plt.title('img2_crop with noise histogram')
plt.show()


# %% 4
# Function performing 5x5 Gaussian filtering
def my_Gauss_filter(noisy_img, my_gauss_kernel):
    # Detect if image is with enough size
    if noisy_img.shape[0] < 5 or noisy_img.shape[1] < 5:
        raise Exception('Need image greater than 5x5')
    # Initialize a array of zeros with the size after filtering: (x-4,y-4)
    ans = np.zeros((noisy_img.shape[0] - 4, noisy_img.shape[1] - 4))
    # Iterate through the original image
    for i in range(noisy_img.shape[0] - 4):
        for j in range(noisy_img.shape[1] - 4):
            # Apply the kernel to the selected part of image with convolution
            # operation and update the answer image
            ans[i, j] = np.sum(noisy_img[i:i + 5, j:j + 5] * my_gauss_kernel)
    return ans


# %% 5
# Generate a 5x5 Gaussian kernel with sigma = 1
gaussian_filter_1D = cv2.getGaussianKernel(5, 1)
gaussian_filter_2D = gaussian_filter_1D * gaussian_filter_1D.T
# Apply the filter to the image with noise
smoothed_img = my_Gauss_filter(img2_with_noise, gaussian_filter_2D).astype(np.uint8)
# Show the smoothed image
plt.imshow(smoothed_img, cmap='gray')
plt.title('smoothed image')
plt.show()

# Try different sigma values
for i in [0.1, 0.5, 1, 5, 10, 50, 100]:
    gaussian_filter_1D = cv2.getGaussianKernel(5, i)
    gaussian_filter_2D = gaussian_filter_1D * gaussian_filter_1D.T
    smoothed_img = my_Gauss_filter(img2_with_noise, gaussian_filter_2D).astype(np.uint8)
    plt.imshow(smoothed_img, cmap='gray')
    plt.title(f'sigma = {i}')
    plt.show()

#%% 6
# Compare the result with the inbuilt function
smoothed_img_inbuilt = cv2.GaussianBlur(img2_with_noise, (5, 5), 1)
plt.imshow(smoothed_img_inbuilt, cmap='gray')
plt.title(f'inbuilt')
plt.show()
gaussian_filter_1D = cv2.getGaussianKernel(5, 1)
gaussian_filter_2D = gaussian_filter_1D * gaussian_filter_1D.T
smoothed_img = my_Gauss_filter(img2_with_noise, gaussian_filter_2D).astype(np.uint8)
identical_pix_count = np.count_nonzero(smoothed_img == smoothed_img_inbuilt[2:-2,2:-2])
print(f'Ratio of identical pixels: {identical_pix_count/508/508}')