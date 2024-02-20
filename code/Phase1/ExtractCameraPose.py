"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Krunal M Bhatt
Date - 02/18/2024
Title: Estimating the Fundamental Matrix
This file contains the code to estimate the fundamental matrix 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from EstimateFundamentalMatrix import *
from utils import *

def cameraPose(E):
    """
    This function will calculate 4 camera pose from the Essential matrix
    Input : Essential matrix (3x3)
    Output : Rotation matrix [4,; 3x1] and translation vector [4; 3x3]
    """
    U_E, S_E, V_E = np.linalg.svd(E)

    #Estimate 4 camera translation from the Essential matrix
    U = U_E[:, 2]
    Cs = [U, -U, U, -U]

    #Estimating the rotation matrix for the 4 camera poses using the Essential matrix
    S = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R1 = U_E @ S @ V_E
    R1 = getRotationMatrix(R1)
    #print('R1: ', R1)
    
    R2 = U_E @ S.T @ V_E
    R2 = getRotationMatrix(R2)
    #print('R2: ', R2)

    Rs = [R1, R1, R2, R2]
    #   print('Rs: ', Rs)

    Final_R = []
    Final_C = []
    e = 1e-2

    #Checking the determinant of the rotation matrix for the correct orientation of the camera
    for C, R in zip(Cs, Rs):
        det = np.linalg.det(R)
        if -1 - e < det < -1 + e:                        #Checking the determinant of the rotation matrix
            R = -R
            C = -C
            Final_R.append(R)
            Final_C.append(C)
        else:
            Final_C.append(C)
            Final_R.append(R)

    return Final_R, Final_C