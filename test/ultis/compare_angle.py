from components.match_pattern import match_pattern

def compare_angle(point,minus_sub_angles,plus_sub_angles, gray_img, template_gray, bboxes, angle, method,result_queue ):
    minus_pointer, plus_pointer = 0,0
    exactly_minus, exactly_plus = 0, 0
    high_point_minus, high_point_plus = point, point
    bestAngle,bestPoint = 0,0

    while minus_pointer < len(minus_sub_angles)  or plus_pointer < len(plus_sub_angles):
        point_minus = match_pattern(gray_img, template_gray, bboxes, minus_sub_angles[minus_pointer], method)
        point_plus = match_pattern(gray_img, template_gray, bboxes, plus_sub_angles[plus_pointer], method)
        if point_minus >= high_point_minus:
            if minus_sub_angles[minus_pointer] >= 360:
                exactly_minus = minus_sub_angles[minus_pointer] - 360
            else:
                exactly_minus = minus_sub_angles[minus_pointer]
            if point_minus > bestPoint:
                bestAngle = exactly_minus
                bestPoint = point_minus
            high_point_minus = point_minus
        
        if point_plus >= high_point_plus:
            if plus_sub_angles[plus_pointer] >= 360:
                exactly_plus = plus_sub_angles[plus_pointer] - 360
            else:
                exactly_plus = plus_sub_angles[minus_pointer]
            if point_plus > bestPoint:
                bestAngle = exactly_plus
                bestPoint = point_plus
            high_point_plus = point_plus
        
        minus_pointer = minus_pointer + 1
        plus_pointer = plus_pointer + 1
        # print("---------------------------------------")
    if bestPoint < point:
        bestAngle = angle
    result_queue.append((bestAngle, bestPoint))
    return bestAngle, bestPoint

#kiểm tra từng phần tử minus_sub_angles,plus_sub_angles, lấy angle mà có số điểm cao nhất 

