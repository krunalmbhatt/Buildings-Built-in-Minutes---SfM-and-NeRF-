"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Krunal M Bhatt
Date - 02/18/2024
Title: Estimating the Fundamental Matrix
This file contains the code to estimate the fundamental matrix 
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from EstimateFundamentalMatrix import *
import random
from utils import *

def getInliersRANSAC(v1, v2, iterations, threshold):
    """
    This function will use RANSAC to get the inliers from the set of matching points in two images
    Input : v1 and v2 non-homogenous feature coordinates 
    Iterations : Number of iterations for RANSAC
    Threshold: error threshold 
    Output : Inliers from the set of matching points in two images
    """
    #Homoegenize the points
    # v1 = np.hstack((v1, np.ones((v1.shape[0], 1))))
    v1 = np.concatenate((v1,np.ones((v1.shape[0],1))),axis=1)  #Nx3
    v2 = np.concatenate((v2,np.ones((v2.shape[0],1))),axis=1)  #Nx3
    # print('v1: ', v1)
    # print('v2: ', v2)

    max_index = []
    for i in range(iterations):
        random_point = random.sample(range(v1.shape[0]-1), 8) #Randomly select 8 points
        v1_random = v1[random_point, :]
        v2_random = v2[random_point, :]
        # print('v1_random: ', v1_random)
        # print('v2_random: ', v2_random)

        _, F = estimateFundamentalMatrix(v1_random, v2_random) #Estimate the fundamental matrix using the 8 points
        err = F @ v1.T  # 3x3 * 3xN = 3xN
        err = err.T 
        err = np.multiply(err, v2) #Nx3
        err = np.sum(err, axis = 1) #Nx1
        # print('err: ', err)
 
        inliers_index = np.abs(err) < threshold  #Nx1
        if np.sum(inliers_index) > np.sum(max_index):
            max_index = inliers_index
            #print('max_index: ', max_index)
    return max_index
