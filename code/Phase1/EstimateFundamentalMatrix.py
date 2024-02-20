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
import math


def estimateFundamentalMatrix(v1, v2):
    """
    In this function, we will use two vectors of the same points in two images to estimate the fundamental matrix
    
    :Inputs: V1 and V2 are two vectors of the same points in two images
    :return: The fundamental matrix 
    i.e.; x_2.T * F * x_1.T = 0
    """

    x_1, y_1 = v1[:, 0], v1[:, 1] # Extracting the x and y coordinates of the points
    x_2, y_2 = v2[:, 0], v2[:, 1] # Extracting the x and y coordinates of the points

    first = np.ones(v1.shape[0])                                                             #We will take the last row for set of point correspondences as 0

    # print('first: ', first)
    
    #Null space of A (Ax = 0)
    A = np.asarray([x_1 * x_2, y_1 * x_2, x_2, x_1 * y_2, y_1 * y_2, y_2, x_1, y_1, first])  # Getting Nx9 A matrix for the equation
    A = np.transpose(A)                                                                      # transpose so matrix is [m x 9], m = matched point pairs
    # print('A: ', A)
    # print('A shape: ', A.shape)
    
    # Now we will solve for the fundamental matrix using SVD
    U, Sigma, V = np.linalg.svd(A)                          # Singular Value Decomposition (NxN, Nx9, 9x9 matrices)
    F = V[np.argmin(Sigma),:].reshape(3, 3)                 # reshape the last row of V to get the fundamental matrix (9x1 to 3x3)

    #Reestimating the fundamental matrix
    U_F, S_F, V_F = np.linalg.svd(F)
    S_F[2] = 0                                              #Enforcing the rank 2 constraint
    diagonal_S_F = np.diag(S_F)                             #Creating a diagonal matrix from the singular values
    reestimated_F = U_F @ diagonal_S_F @ V_F                #Reestimating the fundamental matrix
    return F, reestimated_F

def fundamentalMatrix(i,j,sfm_map):
    """
    This function will calculate the fundamental matrix for the given pair of images
    Input : i and j are the pair of images
          : sfm_map is the dictionary containing the camera pose and the 3D points {i,j} <-[v1,v2]
    Output : Fundamental matrix
    """
    key = (i,j)
    v1, v2, _ = sfm_map.get_feature_matches(key)
    _, F = estimateFundamentalMatrix(v1, v2) 
    return F

def epipolarPoints(F, homogenous = False):
    """
    This function will calculate the epipolar points for matching points in two images
    Input : Fundamental Matrix
    Output : Epipole for image 1 and image 2
    """
    U, S, V = np.linalg.svd(F)                          # Singular Value Decomposition
    e1 = V[2, :]
    #print('e1: ', e1)
    e1 = e1 / e1[2]                                    #Normalizing the epipole

    e2 = U[:, 2]
    #print('e2: ', e2)
    e2 = e2 / e2[2]                                    #Normalizing the epipole
    #print('e1_afternorm: ', e1)
    #print('e2_afternorm: ', e2)

    if not homogenous:
        e1 = e1[0:2]                                 #Returning the epipole in non-homogenous form
        e2 = e2[0:2]                           
    return e1, e2

def epipolarLines(F, v1, v2):
    """
    This function will calculate the epipolar lines for matching points in two images
    Input : Fundamental Matrix and the matching points in two images
    Output : Epipolar lines for image 1 and image 2
    """

    epi_line_1 = np.matmul(F.T, v2.T)                  #Epipolar line for image 1
    epi_line_2 = np.matmul(F, v1.T)                    #Epipolar line for image 2
    return epi_line_1, epi_line_2