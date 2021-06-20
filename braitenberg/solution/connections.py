from typing import Tuple

import numpy as np
import math


def get_motor_left_matrix(shape: Tuple[int, int]) -> np.ndarray:
    x = np.linspace(0, 1, shape[1]//2)
    y = np.linspace(0, 1, shape[0]//2)
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2
    
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    res[math.ceil(res.shape[0]/2):, :gradient.shape[1]] = gradient
    
    x = np.linspace(-1, 0, shape[1]//2)
    y = np.linspace(0, -1, shape[0]//2)
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2

    res[math.ceil(res.shape[0]/2):, math.ceil(res.shape[1]/2):] = gradient

    res[:, :math.ceil(res.shape[1]/4)] = 0
    res[:, -math.ceil(res.shape[1]/4):] = 0
    
    return res


def get_motor_right_matrix(shape: Tuple[int, int]) -> np.ndarray:
    x = np.linspace(0, -1, shape[1]//2)
    y = np.linspace(0, -1, shape[0]//2)
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2
    
    res = np.zeros(shape=shape, dtype="float32")  # write your function instead of this one
    res[math.ceil(res.shape[0]/2):, :gradient.shape[1]] = gradient
    
    x = np.linspace(1, 0, shape[1]//2)
    y = np.linspace(0, 1, shape[0]//2)
    
    a, b = np.meshgrid(x,y)
    gradient = (a+b)/2
    
    res[math.ceil(res.shape[0]/2):, math.ceil(res.shape[1]/2):] = gradient
    
    res[:, :math.ceil(res.shape[1]/4)] = 0
    res[:, -math.ceil(res.shape[1]/4):] = 0
    
    return res
