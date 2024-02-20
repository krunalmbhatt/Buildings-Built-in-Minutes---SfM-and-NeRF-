import cv2
import numpy as np
import csv
import glob

def plot1(img, features, color=(0, 255, 0), marker_type=cv2.MARKER_CROSS, thickness=4):
    """
    input:
        img: image on which features have to be plotted
        features: list of features
        color: color of the feature
        marker: shape of the feature
        thickness: thickness of the feature
    output:
        img: image with features plotted
    """
    copy = img
    for i in features:
        cv2.drawMarker(copy, [int(i[0]), int(i[1])], color, marker_type, thickness)


def getRotationMatrix(M):
    """
    This function will calculate the rotation matrix from the Essential matrix
    Input : Essential matrix
    Output : Rotation matrix
    """
    U_R, S_R, V_R = np.linalg.svd(M)              # Singular Value Decomposition
    S_R = [1,1,1]
    R = U_R @ np.diag(S_R) @ V_R
    #print('R: ', R)
    return R

def load_img(path):
    """
    input:
        path 
    output:
        images - list of images
    """
    extension = ".png"
    img_file = glob.glob(f"{path}/*{extension}",recursive=False)

    img_names = []
    for img in img_file:
        img_name = img.rsplit(".", 1)[0][-1]
        img_names.append(img_name)

    imgs = {int(img_name) : cv2.imread(img_file) for img_file, img_name in zip(img_file,img_names)}
    return imgs, img_names

def load_camera_intrinsics(path):
    """
    input:
        path - location of calibration.txt
    output:
        camera intrinsic matrix
        k=  [[alpha gamma u0 ],
            [  0   beta  v0 ],
            [   0     0    1]] 
    """
    K = []
    with open(path) as file:
        reader = csv.reader(file, delimiter=' ')
        for row in reader:
            K.append([float(row[i]) for i in range(3)])
    K = np.array(K)
    return K


def skew(x):
    """
    This function will calculate the skew symmetric matrix from the given vector
    """
    return np.array([[ 0  , -x[2],  x[1]],
                     [x[2],    0 , -x[0]],
                     [-x[1], x[0],    0]])


########################
# Visualizations TO BE EDITED. I'll edit and push again.
########################
# def show_sample_matches_epipolars(imgs, test_key, matches):

#     img1 = imgs[test_key[0]]
#     img2 = imgs[test_key[1]]
#     v1, v2, _ = matches

#     random.seed(42)
#     inliers = random.sample(range(v1.shape[0]-1),8)
#     v1_sample = v1[inliers, :]
#     v2_sample = v2[inliers, :]

#     show_matches2(img1, img2,[v1_sample, v2_sample], f"test_matches_{test_key}")

#     F, reestimatedF = estimate_fundamental_matrix(v1_sample,v2_sample)

#     # plot the epipolar lines without rank 2 F
#     # expected output: all epipolar do not intersect at one point (or no epipole)
#     show_epipolars(img1, img2, F, [v1_sample, v2_sample], f"test_wo_rank2_{test_key}")

#     # plot the epipolar lines with reestimated F
#     # expected output: all epipolar intersect at one point
#     show_epipolars(img1, img2, reestimatedF, [v1_sample, v2_sample], f"test_w_rank2_{test_key}")

# def show_before_after_RANSAC(imgs, test_key, matches_before, matches_after):

#     img1 = imgs[test_key[0]]
#     img2 = imgs[test_key[1]]
#     v1, v2, _ = matches_before
#     v1_corrected, v2_corrected, _ = matches_after

#     show_matches2(img1, img2,[v1, v2], f"before_RANSAC_{test_key}")
#     show_matches2(img1, img2,[v1_corrected, v2_corrected], f"after_RANSAC_{test_key}")

# def show_disambiguated_and_corrected_poses(Xs_all_poses, X_linear, X_non_linear, C):
#     plt.figure("camera disambiguation")
#     colors = ['red','brown','greenyellow','teal']
#     for color, X_c in zip(colors, Xs_all_poses):
#         plt.scatter(X_c[:,0],X_c[:,2],color=color,marker='.')

#     plt.figure("linear triangulation")
#     plt.scatter(X_linear[:, 0], X_linear[:, 2], color='skyblue', marker='.')
#     plt.scatter(0, 0, marker='^', s=20)

#     plt.figure("nonlinear triangulation")
#     plt.scatter(X_non_linear[:, 0], X_non_linear[:, 2], color='red', marker='x')
#     plt.scatter(C[0], C[1], marker='^', s=20)