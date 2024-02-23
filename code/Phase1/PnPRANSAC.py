"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: PnPRANSAC
This file contains the code for PnP with RANSAC 
"""
import numpy as np
from LinearPnP import linear_pnp
from utils.helpers import homogenize_coords
import random

random.seed(42)

def PnP_RANSAC(features, world_points, K, max_iters=1000, threshold=1e-10):
    # Homogenize world points once before the loop
    #print(world_points)
    world_points_h = homogenize_coords(world_points)
    
    # Store the best inliers as a boolean mask
    u, v = features.T
    maximum_inliers = []
    for i in range(max_iters):
        # Randomly choose 6 correspondences
        indices = random.sample(range(features.shape[0]), 6)
        sample_features = features[ indices ]
        dummy_points_world = world_points_h[ indices]

        # Estimate camera pose using the selected correspondences
        R, C = linear_pnp(sample_features, dummy_points_world, K)

        # Project world points into the camera
        T = -R @ C.reshape((3, 1))
        P = K @ np.hstack((R, T))
        P1, P2, P3 = P
        
        U_ = np.dot(P1, world_points_h.T)
        print('U_', U_.shape)
        V_ = np.dot(P2, world_points_h.T)
        print('V_', V_.shape)
        W_ = np.dot(P3, world_points_h.T)

        u_reprojected = U_ / W_
        print('u_reprojected', u_reprojected.shape)
        v_reprojected = V_ / W_
        print('v_reprojected', v_reprojected.shape)
        # # Compute image coordinates of the projected points
        # u_reprojected = (P[0] @ world_points_h.T) / (P[2] @ world_points_h.T)
        # v_reprojected = (P[1] @ world_points_h.T) / (P[2] @ world_points_h.T)

        # Calculate squared error in image coordinates
        error = (u - u_reprojected) ** 2 + (v - v_reprojected) ** 2

        # Determine inliers based on the threshold
        inliers = abs(error) < threshold 

        # Update the best model if the current one has more inliers
        if inliers.sum() > maximum_inliers.sum():
            maximum_inliers = inliers

    features_inlier = features[maximum_inliers]
    world_pts_inlier = world_points[maximum_inliers]

    # Re-estimate the pose using all inliers
    
    R, C = linear_pnp(features_inlier, world_pts_inlier, K)

    return R, C, features_inlier, world_pts_inlier

# Usage
# features: Nx2 array of 2D feature points
# world_points: Nx3 array of corresponding 3D world points
# K: 3x3 camera intrinsic matrix
# R, C, inlier_features, inlier_world_points = PnP_RANSAC(features, world_points, K)
