# %%
# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

img3 = cv2.imread('image4.jpg', cv2.IMREAD_UNCHANGED)
img3 = cv2.rotate(img3, cv2.ROTATE_180)
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img3 = cv2.resize(img3, (512, 384))
plt.imshow(img3)
plt.title('original image')
plt.show()


# %% 1
def my_rotation(img, angle):
    if angle > 90 or angle < -90:
        raise Exception('angle should be in [-90,90]')
    # get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -angle, 1)
    # get the rotated image
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
    return rotated_img


plt.imshow(my_rotation(img3, -90))
plt.title('-90 degree')
plt.show()

plt.imshow(my_rotation(img3, -45))
plt.title('-45 degree')
plt.show()

plt.imshow(my_rotation(img3, -15))
plt.title('-15 degree')
plt.show()

plt.imshow(my_rotation(img3, 45))
plt.title('45 degree')
plt.show()

plt.imshow(my_rotation(img3, 90))
plt.title('90 degree')
plt.show()


# %% 2
def my_rotation1(img, angle, forward=True):
    # create the result image
    img1 = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    # detect if the angle is in [-90,90]
    if angle > 90 or angle < -90:
        raise Exception('angle should be in [-90,90]')
    # get the center of the image
    center = np.array([img.shape[1] // 2, img.shape[0] // 2])
    # angle to radian
    pi_degree = math.radians(angle)
    # create the rotation matrix
    if forward:
        warping_matrix = np.array([[math.cos(pi_degree), -math.sin(pi_degree)], [math.sin(pi_degree),
                                                                                 math.cos(pi_degree)]])
    else:
        warping_matrix = np.linalg.inv(np.array([[math.cos(pi_degree), -math.sin(pi_degree)], [math.sin(pi_degree),
                                                                                               math.cos(pi_degree)]]))
    # iterate through index of the source or the result image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # center the coordinate
            coordinate = np.array([j,i]) - center
            # perform the rotation
            coordinate_new = np.dot(warping_matrix, coordinate)
            # generate the new coordinate
            x = int(coordinate_new[0] + center[0])
            y = int(coordinate_new[1] + center[1])
            # check if the new coordinate is in the image
            if img.shape[1] > x >= 0 and img.shape[0] > y >= 0:
                # update the result image
                if forward:
                    img1[y,x] = img[i,j]
                else:
                    img1[i, j] = img[y, x]
    return img1.astype(np.uint8)

for i in [-90,-45,-15,45,90]:
    plt.imshow(my_rotation1(img3, i,True))
    plt.title(str(i)+' degree, forward')
    plt.show()
    plt.imshow(my_rotation1(img3, i, False))
    plt.title(str(i) + ' degree, backward')
    plt.show()

#%% 3
def my_rotation2(img, angle, interpolation):
    if angle > 90 or angle < -90:
        raise Exception('angle should be in [-90,90]')
    # get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), -angle, 1)
    # get the rotated image
    rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]),flags=interpolation)
    return rotated_img

plt.imshow(my_rotation2(img3, -45, cv2.INTER_NEAREST))
plt.title('45 degree, nearest')
plt.show()

plt.imshow(my_rotation2(img3, -45, cv2.INTER_LINEAR))
plt.title('45 degree, linear')
plt.show()