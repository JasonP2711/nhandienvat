
import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_pca(contour):
    # Calculate the centroid
    centroid = np.mean(contour, axis=0)
    centroid_x = centroid[0]
    centroid_y = centroid[1]

    # Calculate the covariance matrix
    covariance_matrix = np.cov((contour[:, 0] - centroid_x).T, (contour[:, 1] - centroid_y).T)

    # Perform PCA
    _, eigenvalues, eigenvectors = cv2.eigen(covariance_matrix, True)

    # Determine the orientation angle
    major_axis = eigenvectors[0]
    orientation_angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    return orientation_angle, orientation_angle+180

def apply_min_area(contour):
    rotated_rect = cv2.minAreaRect(contour)
    #Hàm này được sử dụng để tính toán hình chữ nhật có diện tích nhỏ nhất mà có thể bao quanh một đối
    # tượng được xác định bởi đường viền (contour). Hàm này trả về một cấu trúc dữ liệu gọi là "RotatedRect" (hình chữ nhật
    # đã xoay) chứa thông tin về hình chữ nhật này, bao gồm tọa độ trung tâm, kích thước và góc xoay.
    rect_points = cv2.boxPoints(rotated_rect).astype(int)
    #Sau khi có được thông tin về hình chữ nhật đã xoay, hàm này được sử dụng để lấy tọa độ của bốn đỉnh của 
    #hình chữ nhật. Kết quả là một mảng NumPy chứa các tọa độ của các đỉnh của hình chữ nhật đã xoay.
    edge1 = np.array(rect_points[1]) - np.array(rect_points[0])
    edge2 = np.array(rect_points[2]) - np.array(rect_points[1])
    
    reference = np.array([1, 0])  # Horizontal edge
    #so sanh chonj vecto có độ dài lớn hơn để tính toán góc xoay của hình chữ nhật dựa vào định lý cosin dựa vào reference(reference là một vector chỉ đạo ngang (1, 0).) và used_edge
    if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        used_edge = edge1
        angle = (180.0 / np.pi) * (np.arccos(np.dot(reference, used_edge) / (np.linalg.norm(reference) * np.linalg.norm(used_edge))))

    # if np.linalg.norm(edge2) > np.linalg.norm(edge1)    
    else:
        used_edge = edge2
        angle = (180.0 / np.pi)*(np.arccos(np.dot(reference, used_edge) / (np.linalg.norm(reference) * np.linalg.norm(used_edge))))
        angle = (180 - angle)
        
    return -angle
