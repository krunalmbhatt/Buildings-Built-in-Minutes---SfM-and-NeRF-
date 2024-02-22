import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def create_jacobian_sparsity_pattern(num_cameras, num_points, camera_indices, point_indices):
    m = 2 * len(point_indices)  # Two observations (x, y) per point
    n = 6 * num_cameras + 3 * num_points  # Camera parameters + Point coordinates
    A = lil_matrix((m, n), dtype=int)

    for i, (cam_idx, pt_idx) in enumerate(zip(camera_indices, point_indices)):
        # Block for camera parameters
        A[2 * i:2 * i + 2, cam_idx * 6:cam_idx * 6 + 6] = 1
        # Block for point coordinates
        A[2 * i:2 * i + 2, num_cameras * 6 + pt_idx * 3:num_cameras * 6 + pt_idx * 3 + 3] = 1

    return A



def compute_reprojection_error(params, camera_indices, point_indices, points_2d, K, visibility_matrix):
    """
    Compute the reprojection error for the current parameter estimates.
    """
    num_cameras = visibility_matrix.shape[0]
    num_points = visibility_matrix.shape[1]
    camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
    points_3d = params[num_cameras * 6:].reshape((num_points, 3))
    
    error = []
    for cam_idx, pt_idx in zip(camera_indices, point_indices):
        if visibility_matrix[cam_idx, pt_idx]:
            camera_param = camera_params[cam_idx]
            point_3d = points_3d[pt_idx]
            projected_point = project(point_3d, camera_param, K)
            point_2d = points_2d[cam_idx][pt_idx]  # Assuming points_2d is indexed by [camera][point]
            reprojection_error = projected_point - point_2d
            error.append(reprojection_error.ravel())
    return np.concatenate(error)



def project(point, camera_param, K):
    """
    Project a 3D point onto a 2D image plane using camera parameters.
    """
    # Decompose the camera parameters
    R_vec, t_vec = camera_param[:3], camera_param[3:6]
    R, _ = cv2.Rodrigues(R_vec)
    P = K @ np.hstack((R, t_vec.reshape(-1, 1)))
    
    # Project the 3D point
    point_h = np.hstack((point, 1))
    point_projected = P @ point_h
    point_projected /= point_projected[2]
    return point_projected[:2]

def bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, point_indices, K, visibility_matrix):
    """
    Perform bundle adjustment to refine camera parameters and 3D point estimates.
    """
    # Initial parameter vector (flattened camera parameters and 3D points)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    
    # Define the Jacobian's sparsity pattern
    A = create_jacobian_sparsity_pattern(camera_params.shape[0], points_3d.shape[0], camera_indices, point_indices)
    
    # Perform optimization with the sparsity pattern
    res = least_squares(compute_reprojection_error, x0, jac_sparsity=A, method='trf', verbose=2, x_scale='jac', 
                        args=(camera_indices, point_indices, points_2d, K, visibility_matrix))
    
    # Extract optimized parameters
    num_cameras = camera_params.shape[0]
    optimized_camera_params = res.x[:num_cameras * 6].reshape((num_cameras, 6))
    optimized_points_3d = res.x[num_cameras * 6:].reshape((points_3d.shape[0], 3))
    
    return optimized_camera_params, optimized_points_3d


# Example usage
# Assuming camera_params, points_3d, points_2d, camera_indices, point_indices, K, and visibility_matrix
# are already defined and initialized appropriately
# optimized_camera_params, optimized_points_3d = bundle_adjustment(camera_params, points_3d, points_2d, camera_indices, point_indices, K, visibility_matrix)
