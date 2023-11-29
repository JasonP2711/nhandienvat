import cv2
import numpy as np
from components.match_pattern import match_pattern
def compare_angle(point,minus_sub_angles,plus_sub_angles, gray_img, template_gray, bboxes, angle, method ):
    minus_pointer = 0
    plus_pointer = 0
    exactly_minus = 0
    exactly_plus = 0
    high_point_minus = point
    high_point_plus = point
    bestAngle = 0
    bestPoint = 0

    while minus_pointer < len(minus_sub_angles)  or plus_pointer < len(plus_sub_angles):
        print(" angle: ",minus_sub_angles[minus_pointer],plus_sub_angles[plus_pointer])
        point_minus = match_pattern(gray_img, template_gray, bboxes, minus_sub_angles[minus_pointer], method)
        point_plus = match_pattern(gray_img, template_gray, bboxes, plus_sub_angles[plus_pointer], method)
        print(" point_compare: ",point_minus,point_plus)
        if point_minus >= high_point_minus:
            print("minus:")
            exactly_minus = minus_sub_angles[minus_pointer]
            if point_minus > bestPoint:
                bestAngle = exactly_minus
                bestPoint = point_minus
                print("best angle with best point: ",bestAngle, bestPoint)
            high_point_minus = point_minus
        
        if point_plus >= high_point_plus:
            print("plus:")
            exactly_plus = plus_sub_angles[minus_pointer]
            if point_plus > bestPoint:
                bestAngle = exactly_plus
                bestPoint = point_plus
                print("best angle with best point: ",bestAngle, bestPoint)
            high_point_plus = point_plus
        
        minus_pointer = minus_pointer + 1
        plus_pointer = plus_pointer + 1
        print("---------------------------------------")
    print(f"result: {bestAngle}, score: {bestPoint}")
    return bestAngle, bestPoint

#kiểm tra từng phần tử minus_sub_angles,plus_sub_angles, lấy angle mà có số điểm cao nhất 