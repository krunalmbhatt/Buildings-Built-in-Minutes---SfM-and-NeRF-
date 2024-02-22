import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

def quat_to_mat(quaternion):
    return R.from_quat(quaternion).as_matrix()

def reprojection_error(params, points_3D, points_2D, K):
    C = params[:3]
    quat = params[3:7]
    R = quat_to_mat(quat)

    P = K @ np.hstack((R, -R @ C.reshape(3, 1)))
    points_3D_h = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    points_2D_proj = (P @ points_3D_h.T).T

    # Normalize homogeneous coordinates
    points_2D_proj = points_2D_proj[:, :2] / points_2D_proj[:, 2, np.newaxis]

    # Compute the reprojection error
    error = points_2D - points_2D_proj
    return error.ravel()

def nonlinear_pnp(features, world_points, C_initial, R_initial, K):
    # Convert initial rotation matrix to quaternion
    quat_initial = R.from_matrix(R_initial).as_quat()
    
    # Initial parameters (camera center C and quaternion q)
    initial_params = np.hstack((C_initial, quat_initial))

    # Run the optimization
    result = least_squares(
        reprojection_error, 
        initial_params, 
        args=(world_points, features, K)
    )

    # Extract optimized parameters
    C_optimized = result.x[:3]
    quat_optimized = result.x[3:7]
    R_optimized = quat_to_mat(quat_optimized)

    return C_optimized, R_optimized

# Usage
# Assume we have R_initial and C_initial from LinearPnP
# R_optimized, C_optimized = nonlinear_pnp(features, world_points, K, R_initial, C_initial)
