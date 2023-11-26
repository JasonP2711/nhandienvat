import cv2
import numpy as np
from ultis.rotate_object import rotate_object 
import logging

logger = logging.getLogger(__name__)
# mở rộng khung ảnh một khi tọa độ bbox ở sát khung ảnh, cản trở việc roi đối tượng
def padded_image(img_gray, bboxes, esilon_w,epsilon_h):
   x_start = bboxes[0] - esilon_w
   y_start = bboxes[1] - epsilon_h
   x_end = bboxes[0] + bboxes[2] + esilon_w
   y_end = bboxes[1] + bboxes[3] + epsilon_h
   padded_left = np.min(x_start,0)
   padded_top = np.min(y_start,0)
   padded_right = np.min(img_gray.shape[1] - x_end, 0)
   padded_bottom = np.min(img_gray.shape[0] - y_end,0)

   img_padded = cv2.copyMakeBorder(img_gray,np.abs(padded_top),np.abs(padded_bottom), np.abs(padded_left), np.abs(padded_right),cv2.BORDER_CONSTANT, value=0 )
   cv2.imwrite("img_paded.jpg",img_padded)
   print("top:",padded_top)
   return img_padded, x_start, x_end, y_start, y_end, padded_top, padded_left, padded_bottom, padded_right

def template_matching(img_gray, template_gray, boxes,  x_start, x_end, y_start, y_end, padded_top, padded_left, padded_bottom, padded_right):
   img_roi = img_gray[np.abs(y_start) : np.abs(y_end)  ,np.abs(x_start) : np.abs(x_end)  ]
   cv2.imwrite( "img_roi.jpg",img_roi)
   #cần mở vòng ảnh to hơn để thuận tiện trong việc matching

def match_pattern(img_gray, template_gray, boxes, sub_angle, method, threshold ):
   _,_, w_temp,h_temp = rotate_object(template_gray,sub_angle)
   epsilon_w, epsilon_h = np.abs(boxes[2]-w_temp), np.abs(boxes[3]-h_temp)
   img_padded, x_start, x_end, y_start, y_end, top, left, bottom, right = padded_image(img_gray,boxes, epsilon_w, epsilon_h)
   
   template_matching(img_gray, template_gray, boxes,  x_start, x_end, y_start, y_end, top, left, bottom, right)

   # print("point mark: ", point)
   # return point




# matched_points là một mảng hai chiều chứa các điểm tương tự hoặc giá trị liên quan đến mức độ khớp giữa mẫu và ảnh. Mỗi phần tử trong mảng tương ứng với mức độ khớp tại một vị trí cụ thể trong ảnh.

# Hàm cv2.minMaxLoc được sử dụng để tìm giá trị nhỏ nhất và lớn nhất trong mảng matched_points và cũng trả về vị trí (tọa độ x, y) của giá trị lớn nhất.

# Khi bạn gán kết quả từ cv2.minMaxLoc cho các biến _, max_val, _, và max_loc, bạn có các giá trị sau:

# _ (underscore đầu tiên) là giá trị nhỏ nhất trong mảng matched_points, nhưng bạn không sử dụng nó trong ví dụ này, vì bạn chỉ quan tâm đến giá trị lớn nhất.
# max_val là giá trị lớn nhất trong mảng matched_points. Đây có thể được coi là mức độ khớp tối đa giữa mẫu và vùng quan tâm của ảnh.
# _ (underscore thứ hai) là vị trí (tọa độ x, y) của giá trị nhỏ nhất trong mảng, bạn cũng không sử dụng nó.
# max_loc là vị trí (tọa độ x, y) của giá trị lớn nhất trong mảng matched_points. Nó cho biết vị trí tương ứng trên ảnh gốc (img_gray) mà mẫu có mức độ khớp tối đa.
# Sau dòng mã này, bạn có thể sử dụng giá trị max_val để xem mức độ khớp tối đa và max_loc để biết vị trí tương ứng trên ảnh.