#%%
# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#
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
#visualization
plt.plot(coordinate_2D[:,0],coordinate_2D[:,1],'b*')
plt.title(f'chosen uv coordinates on image {image_name}')
plt.imshow(I)
plt.show()


coordinate_3D = np.array(
    [[0, 21,7],[0,21,21],
     [0,7,7],[0,7,21],
     [7,21,0],[21,21,0],
     [7,7,0],[21,7,0],
     [7,0,7],[21,0,7],
     [7,0,21],[21,0,21]
     ]
)
#####################################################################
def calibrate(im, XYZ, uv):
    # copy the image
    im = im[:]
    # Initialize A
    A = np.zeros((2*len(XYZ),12))
    # Iteratively calculate values in A for each pair of point
    for i in range(len(XYZ)):
        A[2*i,:] = np.array(
            [XYZ[i][0],XYZ[i][1],XYZ[i][2],1,0,0,0,0,-uv[i][0]*XYZ[i][0],
             -uv[i][0]*XYZ[i][1],-uv[i][0]*XYZ[i][2],-uv[i][0]]
        )
        A[2*i+1,:] = np.array(
            [0,0,0,0,XYZ[i][0],XYZ[i][1],XYZ[i][2],1,-uv[i][1]*XYZ[i][0],
             -uv[i][1]*XYZ[i][1],-uv[i][1]*XYZ[i][2],-uv[i][1]]
        )
    # Calculate solution of p
    _,_,p = np.linalg.svd(A)
    p = (p[-1] / np.linalg.norm(p[-1])).reshape(3,4)
    mse = 0
    # Verification
    for i in range(len(XYZ)):
        # generate homogeneous point coordinate
        new_XYZ = np.array([XYZ[i][0],XYZ[i][1],XYZ[i][2],1])
        # predict 2D homogeneous coordinate
        new_uv = np.matmul(p,new_XYZ)
        # scaling
        new_uv = new_uv[:2]/new_uv[2]
        # draw circle for visualization
        cv2.circle(im,(int(new_uv[0]),int(new_uv[1])),3,(255,0,0),-1)
        # accumulate mse
        mse+=np.linalg.norm(uv[i]-new_uv)
    print(f'The means square error between the chosen uv coordinates and the '+
          f'corresponding projected points is: {mse/len(XYZ)}')
    # calculate origin in 2d image
    origin = np.matmul(p,np.array([0,0,0,1]))
    origin = tuple((origin[:2]/origin[2]).astype(int))
    # x axis
    x = np.matmul(p,np.array([40,0,0,1]))
    x = tuple((x[:2]/x[2]).astype(int))
    cv2.line(im, origin, x, (0, 255, 0), thickness=3, lineType=8)
    # y axis
    y = np.matmul(p,np.array([0,40,0,1]))
    y = tuple((y[:2]/y[2]).astype(int))
    cv2.line(im, origin, y, (0, 255, 0), thickness=3, lineType=8)
    # z axis
    z = np.matmul(p,np.array([0,0,40,1]))
    z = tuple((z[:2]/z[2]).astype(int))
    # draw line for visualization
    cv2.line(im, origin, z, (0, 255, 0), thickness=3, lineType=8)
    # draw origin for visualization
    cv2.circle(im,(int(origin[0]),int(origin[1])),3,(0,255,0),-1)

    # show annotated image
    plt.imshow(im)
    plt.title('Visualization of calibration')
    plt.show()
    return p


p = calibrate(I,coordinate_3D,coordinate_2D)
# TBD
# T_norm = np.array(
#     [
#         [im.shape[1]+im.shape[0],0,im.shape[1]/2],
#         [0,im.shape[1]+im.shape[0],im.shape[0]/2],
#         [0,0,1]
#      ]
# )
# S_norm = np.array(
#     []
# )
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
def homography(u2Trans, v2Trans, uBase, vBase):
    H = None
    return H 

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

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

