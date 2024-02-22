import numpy as np
import os
import cv2
import argparse
import glob
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from GetInliersRANSAC import *
import csv
from utils import *
#from utils.visualization_utils import *
from utils.data_utils import *
from EstimateFundamentalMatrix import *
#from GetInlierRANSAC import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from LinearTriangulation import *
from DisambiguateCameraPose import *
from NonlinearTriangulation import *
from LinearPnP import *
from PnPRANSAC import *
# from BundleAdjustment import perform_BundleAdjustment
# from matplotlib import pyplot as plt
#from ShowOutputs import *
from collections import namedtuple


from utils.helpers import homogenize_coords

# def main():
#     base_path = args.basePath
#     calib_file = args.calibrationFile

#     #Load the images
#     image, image_names = load_img(base_path)

#     #Camera calibration matrix
#     # K = camera_intrinsics(f"{base_path}{calib_file}")

import numpy as np
import argparse
from utils import *  # Import all the necessary functions

def main(args):
    base_path = args.basePath
    calibration_file = args.calibrationFile
    image_files = load_images(base_path)  # Load images from the base path
    K = load_camera_intrinsics(calibration_file)  # Load the camera calibration matrix

    # Initialize data structures
    sfm_data = SFMDataStructure(base_path)  # This will handle matches and poses internally
    CameraPose = namedtuple('CameraPose', ['C', 'R'])
    camera_poses = [CameraPose(C=None, R=None)]  # Initialize with the first camera at the origin    structure = {}  # Dictionary to hold the 3D structure

    # Iterate through image pairs and perform SfM pipeline
    for i, img1 in enumerate(image_files[:-1]):
        for j, img2 in enumerate(image_files[i+1:], start=i+1):
            print(f"Processing image pair ({i}, {j})")
            #x1, x2 = sfm_data.get_inlier_matches(img1, img2)
            x1, x2 = getInliersRANSAC(img1, img2,1000,0.5)

        

            if i == 0 and j == 1:  # For the first pair of images
                F = estimateFundamentalMatrix(x1, x2)
                E = essential_matrix_from_fundamental_matrix(F, K)
                Cset, Rset = extract_camera_pose(E)
                Xset = linear_triangulation(K, np.zeros(3), np.eye(3), Cset, Rset, x1, x2)
                C, R = disambiguate_camera_pose(Cset, Rset, Xset)
                X = nonlinear_triangulation(K, np.zeros(3), np.eye(3), C, R, x1, x2, Xset)
                camera_poses.append(CameraPose(C, R))
                sfm_data.add_structure(j, X)
            else:
                # Register new images using PnP and add 3D points to the structure
                C, R, X = sfm_data.register_new_image(img1, img2, camera_poses, K)
                camera_poses.append(CameraPose(C, R))
                sfm_data.add_structure(j, X)

            # Build visibility matrix and perform bundle adjustment periodically
            if len(camera_poses) > 3 and j % 3 == 0:
                visibility_matrix = build_visibility_matrix(sfm_data, len(image_files))
                camera_poses, structure = bundle_adjustment(camera_poses, structure, visibility_matrix, K)

    # Final bundle adjustment with all camera poses and 3D points
    visibility_matrix = build_visibility_matrix(sfm_data, len(image_files))
    camera_poses, structure = bundle_adjustment(camera_poses, structure, visibility_matrix, K)

    # Save or visualize the camera poses and the structure
    visualize_structure(structure)
    visualize_camera_poses(camera_poses)
    print("Structure from Motion pipeline completed.")
    
    
def main(args):
    base_path = args.basePath
    calibration_file = args.calibrationFile
    images, image_names = load_images(base_path)
    K = load_camera_intrinsics(calibration_file)

    sfm_map = SFMDataStructure(base_path)
    camera_poses = [(np.zeros(3), np.eye(3))]  # Initialize first camera pose
    world_points = []

    for i in range(len(image_names) - 1):
        for j in range(i + 1, len(image_names)):
            first_image_name = image_names[i]
            second_image_name = image_names[j]

            img1 = images[int(first_image_name)]
            img2 = images[int(second_image_name)]

            # Assuming sfm_map.get_feature_matches returns feature matches correctly for the image pair
            x1, x2, sample_indices = sfm_map.get_feature_matches((i, j))
            inliers = getInliersRANSAC(x1, x2, iterations=1000, threshold=0.5)

            F, _ = estimateFundamentalMatrix(x1[inliers], x2[inliers])
            E = E_from_F(F, K)
            Cset, Rset = cameraPose(E)
            Xset = [triangulate_points(K, camera_poses[-1][0], camera_poses[-1][1], C, R, x1[inliers], x2[inliers]) for C, R in zip(Cset, Rset)]
            # Example conceptual adjustment before calling disambiguate_camera_pose
            filtered_sample_indices = sample_indices[inliers]  # Ensure sample_indices reflects the RANSAC inliers
            C, R, X, visibility_idxs = disambiguate_camera_pose(Cset, Rset, Xset, filtered_sample_indices)
            #C, R = disambiguate_camera_pose(Cset, Rset, Xset,sample_indices)
            X = refine_triangulated_coords(K, camera_poses[-1][1], camera_poses[-1][0], R, C, x1[inliers], x2[inliers], Xset)

            camera_poses.append((C, R))
            world_points.append(X)  # This might need adjustment based on your SFMDataStructure methods

            if len(camera_poses) > 2:  # Adjust based on your implementation details
                # Update camera poses and world points using bundle adjustment
                camera_poses, world_points = bundle_adjustment(camera_poses, world_points, sfm_map, K)

    # Final bundle adjustment with all camera poses and 3D points
    camera_poses, world_points = bundle_adjustment(camera_poses, world_points, sfm_map, K)

    # Visualization and saving results
    sfm_map.visualize_camera_poses(camera_poses)
    sfm_map.visualize_structure(world_points)
    print("Structure from Motion pipeline completed.")
    
def main(args):
    base_path = args.basePath
    calibration_file = args.calibrationFile
    images, image_names = load_images(base_path)
    K = load_camera_intrinsics(calibration_file)

    sfm_map = SFMDataStructure(base_path)
    camera_poses = [(np.zeros(3), np.eye(3))]  # Correctly initialize the first camera pose
    world_points = []
    all_Xsets = []

    for i in range(len(image_names) - 1):
        for j in range(i + 1, len(image_names)):
            # Load images, get feature matches
            x1, x2, sample_indices = sfm_map.get_feature_matches((i, j))
            inliers = getInliersRANSAC(x1, x2, iterations=1000, threshold=0.5)

            F, _ = estimateFundamentalMatrix(x1[inliers], x2[inliers])
            E = E_from_F(F, K)
            Cset, Rset = cameraPose(E)

            # Triangulate points for each camera pose candidate
            #Xset = [triangulate_points(K, camera_poses[-1][1], camera_poses[-1][0], C, R, x1[inliers], x2[inliers]) for C, R in zip(Cset, Rset)]
            Xset = [triangulate_points(K, camera_poses[-1][0], camera_poses[-1][1], C, R, x1[inliers], x2[inliers]) for C, R in zip(Cset, Rset)]
            
            #plot_initial_triangulation(Xset)

            # Disambiguate camera pose and refine world points
            filtered_sample_indices = sample_indices[inliers]  # Ensure sample_indices reflects the RANSAC inliers
            C, R, X, visibility_idxs = disambiguate_camera_pose(Cset, Rset, Xset, filtered_sample_indices)
            all_Xsets.append(X)
            linear_tri = X
            #C, R, X, _ = disambiguate_camera_pose(Cset, Rset, Xset)
            X = refine_triangulated_coords(K, camera_poses[-1][0], camera_poses[-1][1], C, R, x1[inliers], x2[inliers], X)
            nonlin_tri = X
            Xcomp = [linear_tri,nonlin_tri]
            #plot_initial_triangulation(Xcomp)
            camera_poses.append((C, R))
            world_points.extend(X)  # Consider managing world points and their visibility across views

    #print(all_Xsets)  
    #plot_initial_triangulation(all_Xsets)
    # Save or visualize the results
    #plot_initial_triangulation(world_points)

    #sfm_map.visualize_camera_poses(camera_poses)
    #sfm_map.visualize_structure(world_points)
    print("Structure from Motion pipeline completed.")
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2: SfM and NeRF')
    parser.add_argument('--basePath', default='Data/')
    parser.add_argument('--calibrationFile', default='Data/calibration.txt')
    args = parser.parse_args()
    main(args)
    
