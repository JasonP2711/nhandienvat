import cv2
import numpy as np
from ultis.rotate_template import rotate_template 
import logging

logger = logging.getLogger(__name__)

#cắt ảnh từ img_gray rồi xoay template, nếu template đã xoay có số score tương đồng cao nhất thì đó là góc chuẩn
def match_template(roi, template_gray,method,angle,scale,threshold):
   if len(roi.shape) == 3:
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
   else:
        img_gray = roi
        
   if len(template_gray.shape) == 3:
        template = cv2.cvtColor(template_gray, cv2.COLOR_BGR2GRAY)
   else:
        template = template_gray
   h,w = template.shape
#    if angle==0:
#       mask = np.full((w,h,1),255,dtype=np.uint8)
#       rotated_template = template_gray
#       new_w, new_h = w, h
#    else:
   rotated_template, mask, new_w, new_h = rotate_template(template_gray, angle)
   method = eval(method)
   # print("masks1: ",mask)
   # print("masks2: ",rotated_template)

   if (img_gray.shape[0] < rotated_template.shape[0]) or (img_gray.shape[1] < rotated_template.shape[1]):
        logger.warning(f'img_gray shape: {img_gray.shape}, rotated_template shape: {rotated_template.shape}')
        return
   matched_points = cv2.matchTemplate(img_gray, rotated_template, method, None, mask)
   _, max_val, _, max_loc = cv2.minMaxLoc(matched_points)

   if max_val >= threshold and max_val <= 1.0:
     return [*max_loc, angle, scale, max_val, new_w, new_h]
   #   return max_val
      

def process_roi(img_padded, template_gray, method, sub_angle, threshold,
                        top, left, bottom, right, 
                        x_start, x_end, y_start, y_end):
   roi = img_padded[y_start + abs(top): y_end + abs(top) + abs(bottom),x_start + abs(left):x_end + abs(left) + abs(right)]
   cv2.imwrite("lkjl.jpg",roi)
   point = match_template(roi, template_gray,method,sub_angle,100,threshold)
#    print("point: ",point)
   return point

def padded_image(img_gray, bbox, epsilon_w, epsilon_h):
   x_start,x_end = (bbox[0]-epsilon_w, bbox[0]+bbox[2]+epsilon_w)
   y_start, y_end = (bbox[1]-epsilon_h,bbox[1]+bbox[3]+epsilon_h)
   top = min(y_start, 0)
   left = min(x_start, 0)
   bottom = min(img_gray.shape[0] - y_end, 0)
   right = min(img_gray.shape[1] - x_end, 0)
#    nới rộng ảnh ra
   img_padded = cv2.copyMakeBorder(img_gray, abs(top), abs(bottom), abs(left), abs(right), cv2.BORDER_CONSTANT, value=0)
   return img_padded, x_start, x_end, y_start, y_end, top, left, bottom, right

def match_pattern(img_gray, template_gray, boxes, sub_angle, method, threshold ):
   _,_, w_temp,h_temp = rotate_template(template_gray,sub_angle)
   epsilon_w, epsilon_h = np.abs((boxes[2]-w_temp, boxes[3]-h_temp))
   img_padded, x_start, x_end, y_start, y_end, top, left, bottom, right = padded_image(img_gray,boxes, epsilon_w, epsilon_h)
   
   point = process_roi(img_padded, template_gray, method, sub_angle, threshold,
                        top, left, bottom, right, 
                        x_start, x_end, y_start, y_end)
   print("point mark: ", point)
   return point




# matched_points là một mảng hai chiều chứa các điểm tương tự hoặc giá trị liên quan đến mức độ khớp giữa mẫu và ảnh. Mỗi phần tử trong mảng tương ứng với mức độ khớp tại một vị trí cụ thể trong ảnh.

# Hàm cv2.minMaxLoc được sử dụng để tìm giá trị nhỏ nhất và lớn nhất trong mảng matched_points và cũng trả về vị trí (tọa độ x, y) của giá trị lớn nhất.

# Khi bạn gán kết quả từ cv2.minMaxLoc cho các biến _, max_val, _, và max_loc, bạn có các giá trị sau:

# _ (underscore đầu tiên) là giá trị nhỏ nhất trong mảng matched_points, nhưng bạn không sử dụng nó trong ví dụ này, vì bạn chỉ quan tâm đến giá trị lớn nhất.
# max_val là giá trị lớn nhất trong mảng matched_points. Đây có thể được coi là mức độ khớp tối đa giữa mẫu và vùng quan tâm của ảnh.
# _ (underscore thứ hai) là vị trí (tọa độ x, y) của giá trị nhỏ nhất trong mảng, bạn cũng không sử dụng nó.
# max_loc là vị trí (tọa độ x, y) của giá trị lớn nhất trong mảng matched_points. Nó cho biết vị trí tương ứng trên ảnh gốc (img_gray) mà mẫu có mức độ khớp tối đa.
# Sau dòng mã này, bạn có thể sử dụng giá trị max_val để xem mức độ khớp tối đa và max_loc để biết vị trí tương ứng trên ảnh.