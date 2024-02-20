"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Krunal M Bhatt
Date - 02/15/2024
Title: Estimating the Fundamental Matrix
This file contains the code to estimate the fundamental matrix 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from utils import *

def E_from_F(K, F):
    """
    This function will estimate the Essential matrix from the Fundamental matrix
    input : K - Intrinsic matrix (3x3)
            F - Fundamental matrix (3x3)
            arg - 1 or 2 to choose the correct Essential matrix
    output : Essential matrix (3x3)
    """
    E = K.T @ F @ K
    #DEBUG
    #print('E: ', E)
    U_E, S_E, V_E = np.linalg.svd(E)
    #DEBUG
    #print('U_E: ', U_E)
    #print('S_E: ', S_E)
    #print('V_E: ', V_E)

    # Reestimate E from (1,1,0) singular values
    corrected_S = np.array([1,1,0])
    #print('corrected_S: ', corrected_S)
    reestimated_E = U_E @ np.diag(corrected_S) @ V_E
    #print('reestimated_E: ', reestimated_E)
    return reestimated_E

def plot_epipoles(K, F, E, image1, image2, name):
    e1_F, e2_F = epipolarPoints(F)      # Epipoles for Fundamental matrix
    e1_E, e2_E = epipolarPoints(E)      # Epipoles for Essential matrix
    #DEBUG
    # print('e1_F: ', e1_F)              
    # print('e2_F: ', e2_F)
    # print('e1_E: ', e1_E)
    # print('e2_E: ', e2_E)

    e1_F_reprojected = K @ e1_E        # Reprojecting the epipole to the image plane
    e2_F_reprojected = K @ e2_E        # Reprojecting the epipole to the image plane
    #DEBUG
    # print('e1_F_reprojected: ', e1_F_reprojected)
    # print('e2_F_reprojected: ', e2_F_reprojected)

    image1_copy = image1.copy()    
    plot1(image1_copy, [e1_F[0:2]], color=(255, 0, 0),thickness=20)  # Plotting the epipole
    plot1(image1_copy, [e2_F_reprojected[0:2]])

    image2_copy = image2.copy()
    plot1(image2_copy, [e1_F[0:2]], color=(255, 0, 0),thickness=20)  # Plotting the epipole
    plot1(image2_copy, [e2_F_reprojected[0:2]])

    combined = np.hstack((image1_copy, image2_copy))                 # Combining the images
    plt.imshow(f"{name}",combined)                                   # Displaying the images
