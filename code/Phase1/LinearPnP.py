"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: LinearPnP
This file contains the code for LinearPnP 
"""
import numpy as np

def linear_pnp(features, world_points, K):
    """
    Solve for camera pose (R, C) using a linear PnP approach.

    Parameters:
    - features: N x 2 array of 2D image points.
    - world_points: N x 3 array of corresponding 3D world points.
    - K: 3 x 3 camera intrinsic matrix.

    Returns:
    - R: 3 x 3 rotation matrix.
    - C: 3 x 1 camera center in world coordinates.
    """
    N = world_points.shape[0]
    ones = np.ones((N, 1))
    zeros = np.zeros((N, 1))
    X, Y, Z = world_points.T
    u, v = features.T

    # Construct matrix A for the linear system
    # A1 = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), ones, zeros, zeros, zeros, zeros,
    #                 -u.reshape(-1, 1) * X.reshape(-1, 1), -u.reshape(-1, 1) * Y.reshape(-1, 1), -u.reshape(-1, 1) * Z.reshape(-1, 1), -u.reshape(-1, 1)])
    A1 = np.vstack([X, Y, Z, ones, zeros, zeros, zeros, zeros,
                    -u * X, -u * Y, -u * Z, -u]).T # N x 12
    # A2 = np.hstack([zeros, zeros, zeros, zeros, X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1), ones,
    #                 -v.reshape(-1, 1) * X.reshape(-1, 1), -v.reshape(-1, 1) * Y.reshape(-1, 1), -v.reshape(-1, 1) * Z.reshape(-1, 1), -v.reshape(-1, 1)])
    A2 = np.vstack([zeros, zeros, zeros, zeros, X, Y, Z, ones,
                    -v * X, -v * Y, -v * Z, -v]).T 
    A = np.vstack([A1, A2])

    # Solve the system using SVD
    _, S_A, V_A = np.linalg.svd(A)
    P = V_A[np.argmin(S_A),:].reshape((3, 4))

    # Decompose the projection matrix to get R and T
    M = P[:, :3]  # Extract the left 3x3 submatrix
    R = np.linalg.inv(K) @ M  # Adjust R with the inverse of intrinsic parameters

    # Ensure R is a proper rotation matrix
    UR, SR, VR = np.linalg.svd(R)
    
    scale = SR[0]
    R = UR @ VR

    detR = np.linalg.det(R)
    T = np.linalg.inv(K) @ P[:, 3] / scale

    if detR < 0:
        R = -R
        T = -T
    
    # Compute camera center C in world coordinates
    C = -R.T @ T

    return R, C
