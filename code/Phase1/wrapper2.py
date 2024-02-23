import numpy as np
import cv2
import argparse
from utils.data_utils import load_images, load_camera_intrinsics, SFMDataStructure
from EstimateFundamentalMatrix import estimateFundamentalMatrix
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import E_from_F
from ExtractCameraPose import cameraPose
from LinearTriangulation import triangulate_points
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import refine_triangulated_coords
from LinearPnP import linear_pnp
from PnPRANSAC import PnP_RANSAC
from NonlinearPnP import nonlinear_pnp
from BundleAdjustments import bundle_adjustment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def plot_initial_triangulation(world_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Assuming world_points is a list of points, each point being a list or tuple [x, y, z]
    xs = [p[0] for p in world_points]
    ys = [p[1] for p in world_points]
    zs = [p[2] for p in world_points]

    # Scatter plot
    scatter = ax.scatter(xs, ys, zs)

    # Labeling the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('initial triangulation')

    # Show the plot
    plt.show()

# Call this function after your main function
# plot_initial_triangulation(world_points)


def plot_initial_triangulation(Xsets):
    # Create a new figure
    fig, ax = plt.subplots()

    # Loop over each set of points
    for idx, X in enumerate(Xsets):
        # Assume each X is an array-like of shape (N, 3), where N is the number of points
        xs = [p[0] for p in X]  # X coordinate
        zs = [p[2] for p in X]  # Z coordinate
        ax.scatter(xs, zs, label=f'Pose {idx+1}')

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_title('Initial triangulation with disambiguity')
    ax.legend()
    
    # Show the plot
    plt.show()

    
def main(args):
    # Load images and calibration data
    base_path = args.basePath
    calibration_file = args.calibrationFile
    images, image_names = load_images(base_path)
    K = load_camera_intrinsics(calibration_file)
    
    # Initialize data structures
    sfm_map = SFMDataStructure(base_path)
    camera_poses = []  # List of tuples (C, R)
    world_points = []  # List of 3D points in the world coordinate

    # Process first pair of images
    x1, x2, sample_indices = sfm_map.get_feature_matches((0, 1))
    inliers = getInliersRANSAC(x1, x2, iterations=1000, threshold=0.5)

    # Estimate the fundamental and essential matrices
    F, _ = estimateFundamentalMatrix(x1[inliers], x2[inliers])
    E = E_from_F(F, K)
    Cset, Rset = cameraPose(E)

    # Triangulate points for each camera pose candidate and disambiguate
    Xset = [triangulate_points(K, np.zeros(3), np.eye(3), C, R, x1[inliers], x2[inliers]) for C, R in zip(Cset, Rset)]
    plot_initial_triangulation(Xset)
    
    filtered_sample_indices = sample_indices[inliers]  # Ensure sample_indices reflects the RANSAC inliers
    C, R, X, visibility_idxs = disambiguate_camera_pose(Cset, Rset, Xset, filtered_sample_indices)
    #C, R, X = disambiguate_camera_pose(Cset, Rset, Xset,filtered_sample_indices)
    

    # Refine world points using non-linear triangulation
    X = refine_triangulated_coords(K, np.zeros(3), np.eye(3), C, R, x1[inliers], x2[inliers], X)
    


    # Update the SFM data structure
    camera_poses.append((C, R))  # Add the initial two camera poses
    camera_poses.append((np.zeros(3), np.eye(3)))  # The first camera is always at the origin
    world_points.extend(X)  # Add 3D points to the world points list

    # Process the rest of the images
    for i in range(2, len(image_names)):
        # Get feature matches with the first image (reference image)
        x1, x2, sample_indices = sfm_map.get_feature_matches((0, i))
        # Get inliers from feature matches
        inliers = getInliersRANSAC(x1, x2, iterations=1000, threshold=0.5)
        #img_pts, world_pts = sfm_map.get_2d_to_3d_correspondences(i)

        # Register the i-th image using PnP
        #Cnew, Rnew = PnP_RANSAC(X, x2[inliers], K)
        features_2d = x2[inliers]  # These are the 2D points from the current image
        world_points_3d = X  # These are the 3D world points obtained so far

        #print(world_points_3d)
        # Perform PnP RANSAC
        Rnew, Cnew, inlier_features, inlier_world_points = PnP_RANSAC(features_2d,world_points_3d, K)
        Cnew, Rnew = nonlinear_pnp(X, x2[inliers], Cnew, Rnew, K)
        # R_optimized, C_optimized = nonlinear_pnp(features, world_points, K, R_initial, C_initial)


        # Update camera poses
        camera_poses.append((Cnew, Rnew))

        # Triangulate new points and refine
        Xnew = triangulate_points(K, Cnew, Rnew, camera_poses[0][0], camera_poses[0][1], x1[inliers], x2[inliers])
        Xnew = refine_triangulated_coords(K, Cnew, Rnew, camera_poses[0][0], camera_poses[0][1], x1[inliers], x2[inliers], Xnew)

        # Add new 3D points to the world points list
        world_points.extend(Xnew)

    # At this point, additional steps such as Bundle Adjustment can be applied if needed

    # Visualization or export of the SFM data can be done here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2: SfM and NeRF')
    parser.add_argument('--basePath', default='../data/')
    parser.add_argument('--calibrationFile', default='../data/calibration.txt')
    args = parser.parse_args()
    main(args)
    