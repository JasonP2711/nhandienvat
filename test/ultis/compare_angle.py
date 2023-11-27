import cv2
import numpy as np
from components.match_pattern import match_pattern
def compare_angle(point,minus_sub_angles,plus_sub_angles,minus_length,plus_length, gray_img, template_gray, bboxes, angle, method ):
    minus_pointer, minus_check = 0, False
    plus_pointer, plus_check = 0, False
    sub_minus_points = []
    sub_plus_points = []

    while minus_length > 0 or plus_length > 0:
        point_minus = match_pattern(gray_img, template_gray, bboxes, angle[minus_pointer], method)
        point_plus = match_pattern(gray_img, template_gray, bboxes, angle[plus_pointer], method)

#kiểm tra từng phần tử minus_sub_angles,plus_sub_angles, lấy angle mà có số điểm cao nhất 