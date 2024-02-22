import csv
import glob
from typing import List, Tuple

import cv2
import numpy as np
import pprint
import random


def load_images(path, extn = ".png"):
    """
    input:
        path - location from where files have to be loaded
        extn - file extension
    output:
        images - list of images - N
    """
    img_files = glob.glob(f"{path}/*{extn}",recursive=False)

    img_names = []
    for img_file in img_files:
        img_name = img_file.rsplit(".", 1)[0][-1]
        img_names.append(img_name)

    imgs = {int(img_name) : cv2.imread(img_file) for img_file, img_name in zip(img_files,img_names)}
    return imgs, img_names

def load_camera_intrinsics(path: str) -> List[List]:
    """
    input:
        path - location of calibration.txt
    output:
        camera intrinsic matrix 3 x 3
            | alpha gamma u0 |
        K - |   0   beta  v0 |
            |   0     0    1 |
    """
    K = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            K.append([float(row[i]) for i in range(3)])
    K = np.array(K)
    return K