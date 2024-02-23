"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: NonlinearTriangulation
This file contains the code to disambiguate the camera pose
"""

import numpy as np
from scipy.optimize import least_squares
from utils.helpers import homogenize_coords, unhomogenize_coords

def reprojection_error(v, X, P):
    """
    Calculate the reprojection error for a single world point and its observed image point.
    """
    X_homog = np.append(X, 1)  # Ensure X is in homogeneous coordinates
    v_hat = P @ X_homog
    v_hat /= v_hat[2]  # Convert from homogeneous to Cartesian coordinates
    error = v_hat[:2] - v[:2]  # Consider only the x, y components
    return error ** 2  # Return squared error

def compute_residuals(X, v1, v2, P1, P2, feature_point_index):
    """
    Compute the combined residuals for a world point against two sets of feature points and projection matrices.
    """
    error_1 = reprojection_error(v1[feature_point_index], X, P1)
    error_2 = reprojection_error(v2[feature_point_index], X, P2)
    errors = np.concatenate((error_1, error_2))  # Aggregate errors from both views
    print("Error 1", error_1)
    print("Error 2", error_2)
    return errors

def refine_triangulated_coords(K, C1, R1, C2, R2, v1, v2, X0):
    """
    Refine initial world coordinates by minimizing reprojection errors across two views.
    """
    # Ensure feature points are in homogeneous coordinates
    v1_homog = homogenize_coords(v1)
    v2_homog = homogenize_coords(v2)

    # Construct projection matrices
    print("C1 shape before reshape:", C1.shape)
    T1 = -R1 @ C1.reshape(-1, 1)
    P1 = K @ np.hstack((R1, T1))
    T2 = -R2 @ C2.reshape(-1, 1)
    P2 = K @ np.hstack((R2, T2))

    X_optimized = np.empty_like(X0)  # Initialize the array for optimized coordinates

    for i, X0_i in enumerate(X0):
        # Optimize each point individually
        result = least_squares(
            compute_residuals, 
            X0_i,  # Initial guess for the ith world point
            args=(v1_homog, v2_homog, P1, P2, i),
            method='lm'
        )
        X_optimized[i] = result.x

    return X_optimized

