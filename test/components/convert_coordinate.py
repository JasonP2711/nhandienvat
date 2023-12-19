import numpy as np
import cv2

def convert_coordinate(point_detect):
    center = np.vstack(np.array(point_detect[:, 1]))
    possible_grasp_ratio = np.vstack(np.array(point_detect[:, 2])).flatten()
