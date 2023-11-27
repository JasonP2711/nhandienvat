import cv2
import numpy as np
from components.match_pattern import match_pattern
def compare_angle(point,minus_sub_angles,plus_sub_angles,minus_length,plus_length, gray_img, template_gray, bboxes, angle, method ):
    minus_pointer = 0
    plus_pointer = 0
    exactly_minus =3
    exactly_plus = 3 

    while minus_pointer < len(minus_sub_angles)  or plus_length < len(plus_sub_angles):
        print(" angle: ",minus_sub_angles[minus_pointer],plus_sub_angles[plus_pointer])
        point_minus = match_pattern(gray_img, template_gray, bboxes, minus_sub_angles[minus_pointer], method)
        point_plus = match_pattern(gray_img, template_gray, bboxes, plus_sub_angles[plus_pointer], method)
        print(" point_compare: ",point_minus,point_plus)
        if point_minus > point:
            exactly_minus = minus_sub_angles[minus_pointer]
        if point_plus > point:
            exactly_plus = plus_sub_angles[minus_pointer]  
        minus_pointer = minus_pointer + 1
        plus_pointer = plus_pointer + 1
        print("---------------------------------------")

    print("contant angle: ",exactly_minus,exactly_plus)

#kiểm tra từng phần tử minus_sub_angles,plus_sub_angles, lấy angle mà có số điểm cao nhất 