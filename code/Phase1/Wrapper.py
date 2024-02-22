import numpy as np
import cv2
import argparse
from utils.data_utils import load_images, load_camera_intrinsics, SFMDataStructure
from EstimateFundamentalMatrix import estimateFundamentalMatrix
from GetInliersRANSAC import getInliersRANSAC
from EssentialMatrixFromFundamentalMatrix import E_from_F
from ExtractCameraPose import cameraPose
from LinearTriangulation import  triangulate_points
from DisambiguateCameraPose import disambiguate_camera_pose
from NonlinearTriangulation import refine_triangulated_coords
from LinearPnP import linear_pnp
from PnPRANSAC import PnP_RANSAC
from BundleAdjustments import bundle_adjustment

def main(args):
    base_path = args.basePath
    calibration_file = args.calibrationFile
    images, image_names = load_images(base_path)
    #print(images)
    #print(image_names)
    K = load_camera_intrinsics(calibration_file)

    sfm_map = SFMDataStructure(base_path)
    camera_poses = []
    world_points = []

    for i, first_image_name in enumerate(image_names[:-1]):
        for second_image_name in image_names[i+1:]:
            #print(first_image_name)
            img1 = images[int(first_image_name)]
            img2 = images[int(second_image_name)]
            
            # Get feature matches and perform RANSAC
            pair = (first_image_name, second_image_name)
            pair_int = tuple(map(int, pair))
            #print("HELP")
            #print(pair_int)
            x1, x2, sample_indices = sfm_map.get_feature_matches(pair_int)
            inliers = getInliersRANSAC(x1, x2, iterations=1000, threshold=0.5)
            
            # Use inliers to estimate the fundamental matrix
            F, reestimatedF = estimateFundamentalMatrix(x1[inliers], x2[inliers])
            # print(type(F))
            # print(F)
            # print("THIS IS K")
            # print(type(K))
            # print(K)
            E = E_from_F(F, K)
            
            # Extract camera pose and triangulate points
            Cset, Rset = cameraPose(E)
            print(type(Cset))
            print(type(Rset))

            #Xset = triangulate_points(K, np.zeros(3), np.eye(3), Cset, Rset, x1[inliers], x2[inliers])
            Xset = [triangulate_points(K, camera_poses[-1][0], camera_poses[-1][1], C, R, x1[inliers], x2[inliers]) for C, R in zip(Cset, Rset)]

            
            # Disambiguate camera poses
            C, R = disambiguate_camera_pose(Cset, Rset, Xset)
            
            if i == 0:
                camera_poses.append((np.zeros(3), np.eye(3)))  # Add the first camera pose
            camera_poses.append((C, R))
            
            # Perform nonlinear triangulation
            X = refine_triangulated_coords(K, camera_poses[-2][1], camera_poses[-2][0], R, C, x1[inliers], x2[inliers], Xset)
            
            # Add points to the SFMMap
            sfm_map.add_world_points(X, inliers)
            
            if len(camera_poses) >= 3:  # Perform bundle adjustment after every few iterations
                camera_poses, world_points = bundle_adjustment(camera_poses, world_points, sfm_map, K)

    # Final bundle adjustment with all camera poses and 3D points
    camera_poses, world_points = bundle_adjustment(camera_poses, world_points, sfm_map, K)

    # Save or visualize the camera poses and the 3D structure
    sfm_map.visualize_camera_poses(camera_poses)
    sfm_map.visualize_structure(world_points)
    print("Structure from Motion pipeline completed.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2: SfM and NeRF')
    parser.add_argument('--basePath', default='Data/')
    parser.add_argument('--calibrationFile', default='Data/calibration.txt')
    args = parser.parse_args()
    main(args)
    