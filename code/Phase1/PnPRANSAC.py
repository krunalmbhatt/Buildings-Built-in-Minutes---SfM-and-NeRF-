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

def PnP_RANSAC(features, world_points, K, max_iters=1000, threshold=1e-4):
    # Homogenize world points once before the loop
    #print(world_points)
    world_points_h = homogenize_coords(world_points)
    
    # Store the best inliers as a boolean mask
    best_inliers = np.zeros(len(features), dtype=bool)
    best_pose = None

    for i in range(max_iters):
        # Randomly choose 6 correspondences
        #indices = random.sample(range(features.shape[0]-1), 6)
        indices = np.random.choice(len(world_points), 6, replace=False)
        print(indices)
        chosen_features = features[indices]
        chosen_world_points = world_points[indices]

        # Estimate camera pose using the selected correspondences
        R, C = linear_pnp(chosen_features, chosen_world_points, K)

        # Project world points into the camera
        T = -R @ C.reshape((3, 1))
        P = K @ np.hstack((R, T))
        
        # Compute image coordinates of the projected points
        u_reprojected = (P[0] @ world_points_h.T) / (P[2] @ world_points_h.T)
        v_reprojected = (P[1] @ world_points_h.T) / (P[2] @ world_points_h.T)

        # Calculate squared error in image coordinates
        error = (features[:, 0] - u_reprojected) ** 2 + (features[:, 1] - v_reprojected) ** 2

        # Determine inliers based on the threshold
        inliers = error < (threshold ** 2)

        # Update the best model if the current one has more inliers
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_pose = (R, C)

    # Re-estimate the pose using all inliers
    if best_pose is not None:
        R, C = linear_pnp(features[best_inliers], world_points[best_inliers], K)

    return R, C, features[best_inliers], world_points[best_inliers]

# Usage
# features: Nx2 array of 2D feature points
# world_points: Nx3 array of corresponding 3D world points
# K: 3x3 camera intrinsic matrix
# R, C, inlier_features, inlier_world_points = PnP_RANSAC(features, world_points, K)
