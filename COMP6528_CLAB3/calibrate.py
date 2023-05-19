# %%
# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from vgg_KR_from_P import *
import math
# %%
image_name = 'stereo2012a.jpg'
I = plt.imread(image_name)
# Graphical user interface to get 6 points
# plt.imshow(I)
# uv = plt.ginput(6)
# print(uv)

coordinate_2D = np.array(
    [(266.9577548315077, 210.56202469824882), (187.1578923762347, 253.61950444030265),
     (283.0325472685412, 309.8812779699196), (209.54778184210272, 359.82795447070197),
     (343.88711863731055, 199.08003010036782), (438.6135740698289, 212.284323887931),
     (356.51731269497975, 298.39928337203855), (447.22507001823965, 317.9186741884363),
     (331.25692457964146, 368.43945041911275), (421.9646819029015, 392.5516390746629),
     (261.79085726246126, 422.9789247590476), (357.09141242487374, 455.70260936300843)]
)
# visualization
# plt.plot(coordinate_2D[:,0],coordinate_2D[:,1],'b*')
# plt.title(f'chosen uv coordinates on image {image_name}')
# plt.imshow(I)
# plt.show()

coordinate_3D = np.array(
    [[0, 21, 7], [0, 21, 21],
     [0, 7, 7], [0, 7, 21],
     [7, 21, 0], [21, 21, 0],
     [7, 7, 0], [21, 7, 0],
     [7, 0, 7], [21, 0, 7],
     [7, 0, 21], [21, 0, 21]
     ]
)


#####################################################################
def calibrate(im, XYZ, uv):
    # copy the image
    im = im[:]
    # plot on the image
    plt.imshow(im)
    # add blue star on the image according to uv
    plt.plot(uv[:, 0], uv[:, 1], 'bo')
    # Initialize A
    A = np.zeros((2 * len(XYZ), 12))
    # Iteratively calculate values in A for each pair of point
    for i in range(len(XYZ)):
        A[2 * i, :] = np.array(
            [XYZ[i][0], XYZ[i][1], XYZ[i][2], 1, 0, 0, 0, 0, -uv[i][0] * XYZ[i][0],
             -uv[i][0] * XYZ[i][1], -uv[i][0] * XYZ[i][2], -uv[i][0]]
        )
        A[2 * i + 1, :] = np.array(
            [0, 0, 0, 0, XYZ[i][0], XYZ[i][1], XYZ[i][2], 1, -uv[i][1] * XYZ[i][0],
             -uv[i][1] * XYZ[i][1], -uv[i][1] * XYZ[i][2], -uv[i][1]]
        )
    # Calculate solution of p
    _, _, p = np.linalg.svd(A)
    p = (p[-1] / np.linalg.norm(p[-1])).reshape(3, 4)
    mse = 0
    predicted_coordinate = np.zeros(uv.shape)
    # Verification
    for i in range(len(XYZ)):
        # generate homogeneous point coordinate
        new_XYZ = np.array([XYZ[i][0], XYZ[i][1], XYZ[i][2], 1])
        # predict 2D homogeneous coordinate
        new_uv = np.matmul(p, new_XYZ)
        # scaling
        new_uv = new_uv[:2] / new_uv[2]
        predicted_coordinate[i, 0] = new_uv[0]
        predicted_coordinate[i, 1] = new_uv[1]
        # accumulate mse
        mse += np.linalg.norm(uv[i] - new_uv)
    # plot predicted coordinate with red circle
    mse /= uv.shape[0]
    plt.plot(uv[:, 0], uv[:, 1], 'r+')
    print(f'The means square error between the chosen uv coordinates and the ' +
          f'corresponding projected points is: {mse}')
    # calculate origin in 2d image
    origin = np.matmul(p, np.array([0, 0, 0, 1]))
    origin = tuple((origin[:2] / origin[2]).astype(int))
    # x axis
    x = np.matmul(p, np.array([40, 0, 0, 1]))
    x = tuple((x[:2] / x[2]).astype(int))
    plt.plot([origin[0], x[0]], [origin[1], x[1]], 'go-')
    # y axis
    y = np.matmul(p, np.array([0, 40, 0, 1]))
    y = tuple((y[:2] / y[2]).astype(int))
    plt.plot([origin[0], y[0]], [origin[1], y[1]], 'go-')
    # z axis
    z = np.matmul(p, np.array([0, 0, 40, 1]))
    z = tuple((z[:2] / z[2]).astype(int))
    # draw line for visualization
    plt.plot([origin[0], z[0]], [origin[1], z[1]], 'go-')
    # draw origin for visualization
    plt.plot(origin[0], origin[1], 'ko')

    # show annotated image
    plt.title('Visualization of calibration')
    plt.show()
    return p

def get_focal_length(K):
    fx = K[0, 0]
    fy = K[1, 1]
    gamma = K[0, 1]
    theta = math.atan(-gamma / fy)
    fy_actual = fy * math.cos(theta)
    f = math.sqrt(fx ** 2 + fy_actual ** 2)
    return f

def get_camera_center(R,t):
    return -R.T@t

def get_pitch_angle(R):
    return -math.asin(R[2,0])
#
p = calibrate(I, coordinate_3D, coordinate_2D)
K,R,t = vgg_KR_from_P(p)
t = -R@t
focal_length = get_focal_length(K)
camera_center = get_camera_center(R,t)
pitch_angle = get_pitch_angle(R)
#%% 1.6
# resize image
width = int(I.shape[1] / 3)
height = int(I.shape[0] / 3)
dim = (width, height)
I_resize = cv2.resize(I, dim, interpolation=cv2.INTER_AREA)

plt.imshow(I_resize)
plt.title(f'resized image {image_name}')
plt.show()

uv_resize = coordinate_2D/3
XYZ_resize = coordinate_3D/3

p_resize = calibrate(I_resize, XYZ_resize, uv_resize)
K_resize, R_resize, t_resize = vgg_KR_from_P(p_resize)
t_resize = -R_resize@t_resize
focal_length_resize = get_focal_length(K_resize)
camera_center_resize = get_camera_center(R_resize,t_resize)
pitch_angle_resize = get_pitch_angle(R_resize)

'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% your name, date 
'''

############################################################################
# %%
image_name_left = 'Left.jpg'
image_name_right = 'Right.jpg'

I_l = plt.imread(image_name_left)
I_r = plt.imread(image_name_right)

# Graphical user interface to get 6 points
# plt.imshow(I_l)
# uv = plt.ginput(6)
# print(uv)

coordinate_left = np.array([
    (152.5676062234005, 108.5259241909281), (186.49530553866703, 111.10737957361141),
    (228.9049296827501, 112.21371759476136), (290.1223001863831, 114.05761429667803),
    (158.83685500991714, 184.49446830989427), (194.23967168671697, 192.97639313871088)
])
coordinate_right = np.array([
    (169.1626765406504, 147.9853136119445), (222.26690155585015, 146.87897559079448),
    (275.0023472306665, 146.87897559079448), (339.5387317977494, 146.87897559079448),
    (181.3323947733004, 217.31582960401073), (232.96150242696672, 218.0533882847774)
])

plt.imread(image_name_left)
plt.plot(coordinate_left[:,0],coordinate_left[:,1],'b*')
plt.title(f'chosen uv coordinates on image {image_name_left}')
plt.imshow(I_l)
plt.show()

plt.imread(image_name_right)
plt.plot(coordinate_right[:,0],coordinate_right[:,1],'b*')
plt.title(f'chosen uv coordinates on image {image_name_right}')
plt.imshow(I_r)
plt.show()

def homography(u2Trans, v2Trans, uBase, vBase):
    # initialize matrix A
    A = np.zeros((2 * u2Trans.shape[0], 9))
    # traverse through all points
    for i in range(u2Trans.shape[0]):
        # calculate A according to DLT algorithm
        A[2 * i, :] = [u2Trans[i], v2Trans[i], 1, 0, 0, 0, -u2Trans[i] * uBase[i], -v2Trans[i] * uBase[i], -uBase[i]]
        A[2 * i + 1, :] = [0, 0, 0, u2Trans[i], v2Trans[i], 1, -u2Trans[i] * vBase[i], -v2Trans[i] * vBase[i], -vBase[i]]
    # SVD decomposition to obtain a solution
    _, _, V = np.linalg.svd(A)
    # reshape the solution to 3x3 matrix
    H = (V[-1] / np.linalg.norm(V[-1])).reshape(3, 3)
    return H
# Use left as trans and right as base.
H = homography(coordinate_left[:,0],coordinate_left[:,1],coordinate_right[:,0],coordinate_right[:,1])

# warp the trans image
warped_image = cv2.warpPerspective(I_l, H, (I_r.shape[1], I_r.shape[0]))
plt.imshow(warped_image)
plt.title(f"warped image {image_name_left}")
plt.show()

# the distance between warped points and original points in base
mse=0
for i in range(len(coordinate_left)):
    # generate homogeneous point coordinate
    new_uv_predict = np.array([coordinate_left[i][0], coordinate_left[i][1], 1])
    # predict 2D homogeneous coordinate
    new_uv = np.matmul(H, new_uv_predict)
    # scaling
    new_uv = new_uv[:2] / new_uv[2]
    # accumulate mse
    mse += np.linalg.norm(new_uv - coordinate_right[i])
mse/=coordinate_left.shape[0]
print(f'The means square error between predicted selected trans points and corresponding base points is: {mse}')
'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% your name, date 
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q, r] = np.linalg.qr(A.T)  # numpy has QR decomposition, here we can do it
    # with Q: orthonormal and R: upper triangle. Apply QR
    # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R, Q
