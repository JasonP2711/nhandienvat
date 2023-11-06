from components import *
from ultis import *
from ultralytics import YOLO
from copy import deepcopy

imgLink = r'E:\My_Code\test\dataset_specialItems\train\images\20230922_155021_jpg.rf.c5f23709d3b391cef9bf885a73b6f01b.jpg'
templateLink = r'E:\My_Code\test\template.jpg'
modelLink = r"E:\My_Code\test\runs\segment\train\weights\best.pt"

model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(r'E:\My_Code\NhanDienvat\runs\segment\train\weights\last.pt')  # load a custom model


img = cv2.imread(imgLink)
template = cv2.imread(templateLink)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_size = 640

template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
copy_of_template_gray = deepcopy(template_gray)
params = {"low_clip": 10, "high_clip": 90}
copy_of_template_gray = contrast_stretching(copy_of_template_gray,  params)
_, copy_of_template_gray = cv2.threshold(copy_of_template_gray, 100, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite('intensity_template.jpg', copy_of_template_gray)
intensity_of_template_gray = np.sum(copy_of_template_gray == 0)

object_item = proposal_box_yolo(imgLink,modelLink,img_size)#object_item sẽ gồm list thông tin góc và tọa độ của đường bao
print("in4: ", object_item)

minus_modify_angle = np.arange(-1, -20, -1)
plus_modify_angle = np.arange(1, 20, 1)
good_points = []
for angle,bboxes in object_item:
     (x1, y1, w, h) = bboxes
     x2, y2 = x1 + w, y1 + h
#      if(y1-50 < 0 or x1 - 50 < 0 or y2 + 50 >640 or x2 +50 > 640):
#              print("khung hinh to")
#              continue

     center_obj,possible_grasp_ratio = find_center(bboxes, gray_img, intensity_of_template_gray)
     cv2.circle(img,(int(center_obj[0]),int(center_obj[1])),2,(0, 0, 255) ,-1)
     
     if possible_grasp_ratio < 50:
                print("score<50!")
                continue
     print("score: ",possible_grasp_ratio )
     print("angle: ", angle)
     print("tam: ", center_obj)
     minus_sub_angles = angle + minus_modify_angle
     plus_sub_angles = angle + plus_modify_angle
     minus_length = len(minus_sub_angles)
     plus_length = len(plus_sub_angles)
     minus_pointer, minus_check = 0, False
     plus_pointer, plus_check = 0, False
     sub_minus_points = []
     sub_plus_points = []
     method = "cv2.TM_CCORR_NORMED"
     threshold = 0.95
     point = match_pattern(gray_img, template_gray, bboxes, angle, method, threshold)
     print("zzz: ",point)
     cv2.putText(img, (f"({int(center_obj[0])},{int(center_obj[1])})"),(int(center_obj[0])+10,int(center_obj[1]+10)) , cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255), 2)
     cv2.putText(img, (f"({round(angle,2)})"),(int(center_obj[0])+10,int(center_obj[1]+20)) , cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 150, 255), 2)

     minus_sub_angles = angle + minus_modify_angle
     plus_sub_angles = angle + plus_modify_angle
     minus_length = len(minus_sub_angles)
     plus_length = len(plus_sub_angles)
        
     minus_pointer, minus_check = 0, False
     plus_pointer, plus_check = 0, False
     sub_minus_points = []
     sub_plus_points = []
        
     if point is None:
         continue
        
     while True:
                if (minus_length == 0 and plus_length == 0):
                        break

                if minus_length == 0 or minus_pointer >= minus_length:
                        minus_check = True
                elif plus_length == 0 or plus_pointer >= plus_length:
                        plus_check = True

                if not minus_check and minus_length != 0:
                        print("---")
                        minus_point = match_pattern(gray_img, template_gray, bboxes, minus_sub_angles[minus_pointer], method, threshold)
                        if minus_point is not None:
                          minus_check = minus_point[4] < point[4] if minus_pointer == 0 else minus_point[4] < sub_minus_points[-1][4]
                        else:
                          minus_check = True
                        
                        if not minus_check:
                          sub_minus_points.append(minus_point)
                        minus_pointer += 1
                
                if not plus_check and plus_length != 0:
                        print("+++")
                        plus_point = match_pattern(gray_img, template_gray, bboxes, plus_sub_angles[plus_pointer], method, threshold)
                        if plus_point is not None:
                          plus_check = plus_point[4] < point[4] if plus_pointer == 0 else plus_point[4] < sub_plus_points[-1][4]
                        else:
                          plus_check = True
                        
                        if not plus_check:
                          sub_plus_points.append(plus_point)
                        plus_pointer += 1
                
                if minus_check and plus_check:
                 break
        
     best_minus_point = sub_minus_points[-1] if sub_minus_points else None
     best_plus_point = sub_plus_points[-1] if sub_plus_points else None
        
     if (best_minus_point is not None) and (best_plus_point is not None):
         best_point = best_minus_point if best_minus_point[4] > best_plus_point[4] else best_plus_point
     elif (best_minus_point is None) and (best_plus_point is None):
         best_point = point
     else:
         best_point = best_minus_point or best_plus_point
        
     if point:
         good_points.append((best_point, center_obj, possible_grasp_ratio))
     print("good point: ",good_points)
# cv2.drawContours(img, contours, -1, (255, 0, 255), 2)

cv2.imwrite("amTam.jpg",img)
# cv2.imwrite("amTam2.jpg",img2)
