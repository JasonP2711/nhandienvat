import numpy as np
import cv2
import os

def create_homography(point_pixel, point_robot,calib_path):
    point_pixel_homography = np.hstack((point_pixel,np.ones((point_pixel.shape[0],1))))
    transformation_matrix, _ = np.linalg.lstsq(point_pixel_homography, point_robot, rcond=None)[:2]
    homography_matrix = transformation_matrix.reshape(3, 3)
    if not os.path.exists(calib_path):
            os.makedirs(calib_path)
            
    transformation_matrix_path = os.path.join(calib_path, "transformation_matrix.npy")
    np.save(transformation_matrix_path, homography_matrix)
