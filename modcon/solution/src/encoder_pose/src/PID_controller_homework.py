#!/usr/bin/env python
# coding: utf-8

# In[116]:


import numpy as np

# Lateral control

# TODO: write the PID controller using what you've learned in the previous activities

# Note: y_hat will be calculated based on your DeltaPhi() and poseEstimate() functions written previously 

def PIDController(
    v_0, # assume given (by the scenario)
    y_ref, # assume given (by the scenario)
    y_hat, # assume given (by the odometry)
    prev_e_y, # assume given (by the previous iteration of this function)
    prev_int_y, # assume given (by the previous iteration of this function)
    delta_t): # assume given (by the simulator)
    """
    Args:
        v_0 (:double:) linear Duckiebot speed.
        y_ref (:double:) reference lateral pose
        y_hat (:double:) the current estiamted pose along y.
        prev_e_y (:double:) tracking error at previous iteration.
        prev_int_y (:double:) previous integral error term.
        delta_t (:double:) time interval since last call.
    returns:
        v_0 (:double:) linear velocity of the Duckiebot 
        omega (:double:) angular velocity of the Duckiebot
        e_y (:double:) current tracking error (automatically becomes prev_e_y at next iteration).
        e_int_y (:double:) current integral error (automatically becomes prev_int_y at next iteration).
    """
    # TODO: these are random values, you have to implement your own PID controller in here
    # Tracking error
    e_y = y_ref - y_hat

    # integral of the error
    e_int_y = prev_int_y + e_y*delta_t

    # anti-windup - preventing the integral error from growing too much
    e_int_y = max(min(e_int_y,1),-1)
    
    e_der = (e_y - prev_e_y)/delta_t
    
    k_p = 8
    k_d = 10
    k_i = 0.005
    
    if abs(e_y) <= 0.07:
        k_d *= 4
        if abs(e_der) >= 0.008:
            k_d *= 4
            
    omega = k_p*e_y + k_i*e_int_y + k_d*e_der
    
    print()
    print("k_d  : ", k_d)
    print("y_ref: ", y_ref)
    print("e    : ", e_y)
    print("e_der: ", e_der)
    print("e_int: ", e_int_y)
    print("OMEGA: ", omega)
    print()
    print()
    return [v_0, omega], e_y, e_int_y

