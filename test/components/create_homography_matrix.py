import numpy as np
import cv2
import os

def create_homography(point_pixel, point_robot,calib_path):
    point_pixel_homography = np.hstack((point_pixel,np.ones((point_pixel.shape[0],1))))
    transformation_matrix, _ = np.linalg.lstsq(point_pixel_homography, point_robot, rcond=None)[:2]
    homography_matrix = transformation_matrix.reshape(3, 3)
    #c2 sử dụng cv2:
    # Điểm tương ứng trên hình ảnh nguồn và đích
        # pts_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        # pts_dst = np.array([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])

        # # Ước lượng ma trận homography bằng RANSAC
        # homography_matrix, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

        # # In ma trận homography
        # print("Homography Matrix:")
        # print(H)
    if not os.path.exists(calib_path):
            os.makedirs(calib_path)
            
    transformation_matrix_path = os.path.join(calib_path, "transformation_matrix.npy")
    np.save(transformation_matrix_path, homography_matrix)
