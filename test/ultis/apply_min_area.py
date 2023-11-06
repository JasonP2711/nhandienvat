import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_min_area(contour):
#  img = cv2.imread(r"E:\My_Code\test\dataset_specialItems\train\images\20230922_155002_jpg.rf.90731a3792c399a6fae5458b30975abf.jpg")
 rotated_rect= cv2.minAreaRect(contour)
 #lấy góc mặc định
 center, size, angle_zz = cv2.minAreaRect(contour)
#  print("goc mac dinh: ",angle_zz)
 #
 rect_points = cv2.boxPoints(rotated_rect).astype(int)
 edge1 = np.array(rect_points[1]) - np.array(rect_points[0])
 edge2 = np.array(rect_points[2]) - np.array(rect_points[1])
 reference = np.array([1, 0]) 
#  với ta có adge1 và edge2 là toa độ lần lượt của vecto 01 và vecto 12
#  và cùng với cú pháp numpy np.linalg.norm(edge) ta sẽ tính được độ dài của 2 vecto đó, độ dài nào lớn hơn thì 
#  được dùng làm đoạn thẳng kết hợp với điểm tham chiếu để tính toán góc bằng công thức cosin
 if np.linalg.norm(edge1) > np.linalg.norm(edge2):
        used_edge = edge1
       #  np.arccos dùng để tính phép tính arccos trong lượng giác, np.dot dùng để tính tích vô hướng
        angle = (180.0 / np.pi) * (np.arccos(np.dot(reference, used_edge) / (np.linalg.norm(reference) * np.linalg.norm(used_edge))))
       #  print(angle)
    # if np.linalg.norm(edge2) > np.linalg.norm(edge1)    
 else:
        used_edge = edge2
        angle = (180.0 / np.pi)*(np.arccos(np.dot(reference, used_edge) / (np.linalg.norm(reference) * np.linalg.norm(used_edge))))
        angle = (180 - angle)
       #  print(angle)
 return -angle
