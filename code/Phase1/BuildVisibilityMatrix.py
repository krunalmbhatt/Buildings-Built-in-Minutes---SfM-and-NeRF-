"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: BuildVisibilityMatrix
This file contains the code to build the visibility matrix
"""

import numpy as np

def build_visibility_matrix(X_found, filtered_feature_flag, nCam):
    """
    Constructs the visibility matrix for 3D points and cameras.

    Parameters:
    - X_found: A boolean array indicating whether a 3D point has been triangulated.
    - filtered_feature_flag: A 2D boolean array indicating if a feature is observed in each camera.
    - nCam: The number of cameras.

    Returns:
    - visibility_matrix: A binary matrix where entry (i, j) is 1 if the jth point is visible from the ith camera.
    """

    # Initialize the visibility matrix with zeros
    visibility_matrix = np.zeros((nCam + 1, len(X_found)), dtype=int)

    # Update the visibility matrix based on filtered_feature_flag and X_found
    for i in range(nCam + 1):
        visibility_matrix[i] = X_found & filtered_feature_flag[:, i]

    return visibility_matrix

    # # Example usage
    # X_found = np.array([True, False, True, True])  # Example 3D points status
    # filtered_feature_flag = np.array([
    #     [True, False, True],
    #     [False, True, False],
    #     [True, True, True],
    #     [False, False, True]
    # ])  # Example visibility of features across 3 cameras
    # nCam = 2  # Number of cameras
    # visibility_matrix = build_visibility_matrix(X_found, filtered_feature_flag, nCam)
    # print("Visibility Matrix:\n", visibility_matrix)
