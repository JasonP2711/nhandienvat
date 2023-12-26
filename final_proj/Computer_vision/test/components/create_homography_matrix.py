import numpy as np
import cv2


def create_homography(point_pixel):
    
    # point_pixel = np.array(point_pixel,dtype="object")
    # # print("point pixel: ",point_pixel[:,0])
    # input_pixel = np.vstack(point_pixel[:,0], dtype=float)
    input_pixel = np.array([[1449, 1869], [1919, 1851], [2165, 1822], [1894, 1574]],dtype=float)
    print("input_pixel: ", input_pixel)
    #toa do tung vi tri theo toa do cua robot
    input_rb = np.array([[465.05, 137.18], [442, 168.95], [411.83, 159.71], [441.66, 130.43]],dtype=float)
    # Ước lượng ma trận homography bằng RANSAC
    homography_matrix, _ = cv2.findHomography(input_pixel, input_rb, cv2.RANSAC, 5.0)
    output_file_path = 'homography_matrix.npy'
    np.save(output_file_path, homography_matrix)
    # In ma trận homography
    print("Homography Matrix:")
    print(homography_matrix)

def convert_point(point,input_file_path):
    center = np.array([sublist[0] for sublist in point],dtype=np.float32)
    angle = np.array([sublist[1] for sublist in point],dtype=np.float32)
    center = np.hstack((center, np.ones((center.shape[0], 1)))).T
    # # Nạp ma trận homography từ tệp tin .npy
    homography_matrix = np.load(r'E:\My_Code\NhanDienvat\final_proj\Computer_vision\test\homography_matrix.npy')
    # print("matrix: ",homography_matrix)
    transformed_point = np.dot(homography_matrix, center)
    # Chia cho phần tử cuối cùng để có tọa độ (x, y, 1)
    transformed_point = transformed_point / transformed_point[2]
    center_point = transformed_point[:2].T
    # print("result convert: ",center_point)
    result = np.array(list(zip(center_point[:,0],center_point[:,1], angle)), dtype=np.float32)
    # return transformed_point[0], transformed_point[1]
    return result