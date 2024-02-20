import numpy as np
import cv2
import argparse
import glob
from EstimateFundamentalMatrix import *
from EssentialMatrixFromFundamentalMatrix import *
from ExtractCameraPose import *
from GetInliersRANSAC import *
import csv
from utils import *

def main():
    base_path = args.basePath
    calib_file = args.calibrationFile

    #Load the images
    image, image_names = load_img(base_path)

    #Camera calibration matrix
    # K = camera_intrinsics(f"{base_path}{calib_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 2: SfM and NeRF')
    parser.add_argument('--basePath', default='../data/')
    parser.add_argument('--calibrationFile', default='../data/calibration.txt')
    args = parser.parse_args()
    main(args)