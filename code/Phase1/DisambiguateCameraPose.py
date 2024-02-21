"""
Project 2: Buildings built in minutes - SfM and NeRF
Coded by - Jesulona Akinyele
Date - 02/15/2024
Title: DisambiguateCameraPose
This file contains the code to disambiguate the camera pose
"""

import numpy as np

def disambiguate_camera_pose(Cs, Rs, Xs, orig_idxs):
    """
    Finds the correct camera pose obtained from ExtractCameraPose.py.
    Inputs:
        Cs - List of translations for all possible poses.
        Rs - List of rotations for all possible poses.
        Xs - List of 3D points for all poses.
        orig_idxs - Original indices of points used for triangulation.
    Outputs:
        C - Translation of the correct pose.
        R - Rotation of the correct pose.
        X - 3D points valid under the correct pose.
        visibility_idxs - Indices of valid 3D points.
    """
    max_inliers = []
    for i, (C, R, X) in enumerate(zip(Cs, Rs, Xs)):
        r3 = R[:, 2]  # Third column of R
        C = C.reshape((3, 1))  # Ensure C is a column vector

        # Cheirality condition for each point in X
        cond1 = X[:, 2]  # Z-coordinates of points in world space
        cond2 = np.dot(r3, (X.T - C))  # Projection on camera's viewing direction

        # Identify inliers where both conditions are positive
        inliers = np.logical_and(cond1 > 0, cond2 > 0)

        # Update the best pose if it has more inliers
        if len(max_inliers) == 0 or np.sum(inliers) > np.sum(max_inliers):
            correctC, correctR, correctX = C.flatten(), R, X[inliers]
            visibility_idxs = orig_idxs[inliers]
            max_inliers = inliers

    return correctC, correctR, correctX, visibility_idxs
