"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: Helper Functions
This file contains the code for matrix manipulations
"""
import numpy as np
def homogenize_coords(coords):
    """
    Convert coordinates from non-homogeneous to homogeneous representation.
    N x m -> N x (m+1)
    """
    return np.concatenate((coords, np.ones((coords.shape[0], 1))), axis=1)

def unhomogenize_coords(coords):
    """
    Convert coordinates from homogeneous to non-homogeneous representation.
    N x (m+1) -> N x m
    """
    return coords[:, :-1] / coords[:, [-1]]

def unhomogenize_coords(homog_coords):
    """
    Convert homogeneous coordinates back to 2D by dividing by the last component.
    """
    return homog_coords[:, :-1] / homog_coords[:, -1, np.newaxis]
  
def skew(x):
    """
    3, -> 3 x 3
    """
    return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])


