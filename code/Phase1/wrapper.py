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

from utils.helpers import homogenize_coords

# def main():
#     base_path = args.basePath
#     calib_file = args.calibrationFile

#     #Load the images
#     image, image_names = load_img(base_path)

#     #Camera calibration matrix
#     # K = camera_intrinsics(f"{base_path}{calib_file}")




from utils import *  # Assuming all utility functions and classes are in this package

def structure_from_motion(base_path, calibration_file):
    # Load images and camera calibration
    imgs = load_images(base_path)
    K = load_camera_intrinsics(calibration_file)

    # Initialize the SFM data structure
    sfm_data = SFMDataStructure(imgs, K)

    # Process all possible pairs of images
    for img_pair in sfm_data.get_image_pairs():
        x1, x2 = sfm_data.get_inlier_matches(img_pair)

        if img_pair == (1, 2):  # For the first two images
            F = EstimateFundamentalMatrix(x1, x2)
            E = EssentialMatrixFromFundamentalMatrix(F, K)
            Cset, Rset = ExtractCameraPose(E)
            Xset = LinearTriangulation(K, Cset, Rset, x1, x2)

            C, R = DisambiguateCameraPose(Cset, Rset, Xset)
            X = NonlinearTriangulation(K, C, R, x1, x2, Xset)

            sfm_data.register_initial_pair(img_pair, C, R, X)

        else:  # For the rest of the images
            C, R, X = sfm_data.register_next_image(img_pair)

            # Perform Bundle Adjustment
            Cset, Rset, X = BundleAdjustment(C, R, X, sfm_data)

    # After processing all images, return the result
    return X, Cset, Rset

def main(args):
    base_path = args.basePath
    calibration_file = args.calibrationFile

    # Run the SFM pipeline
    X, Cset, Rset = structure_from_motion(base_path, calibration_file)

    # Visualize or save the results
    visualize_3D_points(X)
    save_camera_poses(Cset, Rset)
    print("Structure from Motion has been completed.")

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2: SfM and NeRF')
    parser.add_argument('--basePath', default='../data/')
    parser.add_argument('--calibrationFile', default='../data/calibration.txt')
    args = parser.parse_args()
    main(args)
    
