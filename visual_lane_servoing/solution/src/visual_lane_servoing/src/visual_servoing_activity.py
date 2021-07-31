#!/usr/bin/env python
# coding: utf-8

# In[18]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np
import math


def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                    using the masked left lane markings (numpy.ndarray)
    """
    steer_matrix_left_lane = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one

    x = np.linspace(0, -1, shape[1]//2)
    y = np.linspace(0, -1, shape[0])
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2
    
    steer_matrix_left_lane[:, :gradient.shape[1]] = gradient
    
#     x = np.linspace(1, 0, shape[1]//2)
#     y = np.linspace(0, 1, shape[0]//2)
    
#     a, b = np.meshgrid(x,y)
#     gradient = (a+b)/2

#     steer_matrix_left_lane[math.ceil(steer_matrix_left_lane.shape[0]/2):, math.ceil(steer_matrix_left_lane.shape[1]/2):] = gradient

    return steer_matrix_left_lane

# In[20]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK


def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control 
                                     using the masked right lane markings (numpy.ndarray)
    """
    steer_matrix_right_lane = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one

#     x = np.linspace(0, -1, shape[1]//2)
#     y = np.linspace(0, -1, shape[0]//2)
    
#     a, b = np.meshgrid(x,y)
#     gradient = (a+b)/2
    
#     steer_matrix_right_lane[math.ceil(steer_matrix_right_lane.shape[0]/2):, :gradient.shape[1]] = gradient
    
    x = np.linspace(3.2, 0, shape[1]//2)
    y = np.linspace(0, 3.2, shape[0])
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2
    
    steer_matrix_right_lane[:, math.ceil(steer_matrix_right_lane.shape[1]/2):] = gradient
    
    return steer_matrix_right_lane

# In[22]:


# The function written in this cell will actually be ran on your robot (sim or real). 
# Put together the steps above and write your DeltaPhi function! 
# DO NOT CHANGE THE NAME OF THIS FUNCTION, INPUTS OR OUTPUTS, OR THINGS WILL BREAK

import cv2
import numpy as np


def detect_lane_markings(image):
    """
        Args:
            image: An image from the robot's camera in the BGR color space (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """
    
    h, w, _ = image.shape
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(0,0), 1)
    
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    
    threshold = 95
    mask_mag = (Gmag > threshold)
    
    mask_left = np.ones(sobelx.shape)
    mask_left[:,int(np.floor(w/2)):w + 1] = 0
    mask_right = np.ones(sobelx.shape)
    mask_right[:,0:int(np.floor(w/2))] = 0
    
    mask_sobelx_pos = (sobelx > 0)
    mask_sobelx_neg = (sobelx < 0)
    mask_sobely_pos = (sobely > 0)
    mask_sobely_neg = (sobely < 0)
    
    white_lower_hsv = np.array([0, 0, 160])         # CHANGE ME
    white_upper_hsv = np.array([179, 50, 255])      # CHANGE ME
    yellow_lower_hsv = np.array([15, 60, 110])      # CHANGE ME
    yellow_upper_hsv = np.array([35, 255, 255])      # CHANGE ME
    
    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)
    
    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white
    
#     max_left = np.max(mask_left_edge)
#     max_right = np.max(mask_right_edge)
#     mask_left_edge[mask_left_edge > 0.6*max_left] = 0
#     mask_right_edge[mask_right_edge > 0.6*max_right] = 0
    
#     mask_left_edge[mask_left_edge > 0] = 1.0
#     mask_right_edge[mask_right_edge > 0] = 1.0
#     print(np.sum(mask_left_edge > 0))
    
    
#     mask_left_edge[mask_left_edge > 0] = 1.0
#     mask_right_edge[mask_right_edge > 0] = 1.0

    # Debug    
    STEER_LEFT_LM = get_steer_matrix_left_lane_markings((h,w))
    STEER_RIGHT_LM = get_steer_matrix_right_lane_markings((h,w))
    steering  = np.sum( STEER_LEFT_LM * mask_left_edge) + np.sum( STEER_RIGHT_LM * mask_right_edge)
    print("my steering: ",steering)
    print("left: ",np.sum( STEER_LEFT_LM * mask_left_edge))
    print("right: ",np.sum( STEER_RIGHT_LM * mask_right_edge))
    
#     print(mask_left_edge[mask_left_edge>0])
#     print(mask_right_edge[mask_right_edge>0])
    
    
    
    return (mask_left_edge, mask_right_edge)

# UnitTestDLM(detect_lane_markings)
