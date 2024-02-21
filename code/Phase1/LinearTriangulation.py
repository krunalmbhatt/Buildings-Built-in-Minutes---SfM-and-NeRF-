"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: LinearTriangulation
This file contains the code to estimate the fundamental matrix 
"""

import numpy as np
from utils.helpers import homogenize_coords, skew


def triangulate_points(K, C1, R1, C2, R2, v1, v2):
    """
    Returns the world coordinates for the features in both images using
    triangulation
    inputs:
        K - camera calibration matrix - 3 x 3
        C1 - 1st camera translation - 3,
        R1 - 1st camera rotation - 3 x 3
        C2 - 2nd camera translation  - 3,
        R2 - 2nd camera rotation - 3 x 3
        v1- N x 2
        v2- N x 2
    outputs:
        world_points: N x 3
    """
    # Convert image points to homogeneous coordinates
    v1_h = homogenize_coords(v1)
    v2_h = homogenize_coords(v2)
    
    # Construct camera projection matrices
    P1 = K @ np.hstack((R1, -R1 @ C1.reshape(3, 1)))
    P2 = K @ np.hstack((R2, -R2 @ C2.reshape(3, 1)))
    
    # Initialize the list to hold the 3D world points
    Xs = []

    # Process each pair of points
    for point1, point2 in zip(v1_h, v2_h):
        # Ccompute the skew-symmetric matrices
        A1 = skew(point1) @ P1 # 3 x 4
        A2 = skew(point2) @ P2 # 3 x 4

        # Stack the equations
        A = np.vstack((A1, A2))  # 6x4

        # Solve the system using SVD
        U, sigma, Vt = np.linalg.svd(A)
        X = Vt[-1]  # Take the last row of V^T (smallest singular value)

        # Normalize to make the last component 1 (homogeneous coordinates)
        X = X / X[-1]

        # Store the 3D point
        Xs.append(X[:3])

    # Convert the list of 3D points into a numpy array
    Xs = np.vstack(Xs)

    return Xs

