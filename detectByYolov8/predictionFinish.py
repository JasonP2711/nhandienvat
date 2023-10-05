
from ultralytics import YOLO
import cv2
import numpy as np

import argparse
from math import atan2, cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

link = r'E:\My_Code\NhanDienvat\detectByYolov8\dataset_specialItems\train\images\20230922_155358_jpg.rf.1aba5a0a66c2b6f2194a1af722b00df7.jpg'

image = cv2.imread(link)

def drawAxis(img, a, b, colour, scale,degree):
 print("type: ",type(a))
 p = list(a)
 q = list(b)
 text1 = f"({p[0]}, {p[1]})"
 
#  cv.putText(img, text1, (p[0]+30,p[1]+30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
 angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
 print("angle: ",degree)
 text2 = f"{round(degree,2)} degree"
 cv2.putText(img, text2, (p[0]+10,p[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

 hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 # Here we lengthen the arrow by a factor of scale
 q[0] = p[0] - scale * hypotenuse * cos(angle)
 q[1] = p[1] - scale * hypotenuse * sin(angle)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 # create the arrow hooks
 p[0] = q[0] + 9 * cos(angle + pi / 4)
 p[1] = q[1] + 9 * sin(angle + pi / 4)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 p[0] = q[0] + 9 * cos(angle - pi / 4)
 p[1] = q[1] + 9 * sin(angle - pi / 4)
 cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), colour, 1, cv2.LINE_AA)
 
def getOrientation(pts, img):
#  print(pts)
 sz = len(pts)
 data_pts = np.empty((sz, 2), dtype=np.float64)
 for i in range(data_pts.shape[0]):
  data_pts[i,0] = pts[i,0,0]
  data_pts[i,1] = pts[i,0,1]
 # Perform PCA analysis
 mean = np.empty((0))
 mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
 # Store the center of the object
 cntr = (int(mean[0,0]), int(mean[0,1]))
 print("center: ",cntr)
 
 cv2.circle(img, cntr, 2, (255, 0, 255), 1)
 p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
 p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
#  cv.circle(img, (eigenvectors[0,1], eigenvectors[0,0]), 2, (0, 0, 255), 1)
 angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
 degree = angle * (180 / np.pi)

 drawAxis(img, cntr, p1, (0, 255, 0), 2,degree)
 drawAxis(img, cntr, p2, (255, 255, 0), 5,degree)
 return degree


# ///////////////////////////////////

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(r'E:\My_Code\NhanDienvat\runs\segment\train\weights\best.pt')  # load a custom model

# Predict with the model
results = model(link, save = True)  # predict on an image


print("àdd: ", np.array(results[0].boxes.conf, dtype="float").round(2))

myList = []

mylistmask = []

checkitem = []
for r in results:
    # print("àdd: ",r.masks.xy)
    myList = r.boxes.xywh.tolist()
    checkitem = r.boxes.cls.tolist()
    print("lít: ",checkitem)

count = 0

while count < len(checkitem):
  print("count: ",count)
  if(checkitem[count] == 3.0):
    print("3")
    bBoxitem = myList[count]
    #tạo một mảng numpy mới có cùng kích thước và kiểu dữ liệu như một mảng có tên là image, nhưng tất
    # cả các giá trị pixel của mảng mới sẽ được đặt thành 0.
    mask = np.zeros_like(image, dtype=np.uint8)
#     # Định nghĩa màu xanh mà bạn muốn sử dụng (ví dụ: màu xanh lá cây)
#     green_color = ( 255,255, 0)  # Xanh lá cây: (B, G, R)

# # Gán giá trị màu xanh cho mặt nạ
#     mask[:, :] = green_color

    a = (bBoxitem[0] - bBoxitem[2]/2, bBoxitem[1] - bBoxitem[3]/2)
    b = (bBoxitem[0] + bBoxitem[2]/2, bBoxitem[1] - bBoxitem[3]/2)
    c = (bBoxitem[0] + bBoxitem[2]/2, bBoxitem[1] + bBoxitem[3]/2)
    d = (bBoxitem[0] - bBoxitem[2]/2, bBoxitem[1] + bBoxitem[3]/2)

    points = np.array([a, b,c ,d], dtype=np.int32)
    cv2.fillPoly(mask, [points], (255, 255, 255))
    src = cv2.bitwise_and(image, mask)
    # cv2.imwrite("abc.jpg",src)
    beta = -60  # Giảm độ sáng
        # Sử dụng hàm cv2.convertScaleAbs để giảm độ sáng của ảnh
    result = cv2.convertScaleAbs(src, alpha=1, beta=beta)
    # Check if image is loaded successfully
    if result is None:
    #  print('Could not open or find the image: ', args.input)
        exit(0)
    # cv2.imshow('src', src)
    # Convert image to grayscale
    gray_picture = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    gray_inverted = cv2.bitwise_not(gray_picture)
    # Convert image to binary
    _, bw = cv2.threshold(gray_picture, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('bibi.jpg', bw)
    
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
    # Calculate the area of each contour
      area = cv2.contourArea(c)
    # Ignore contours that are too small or too large
      if area < 1e2*4 or 1e5 < area:
        continue
    # Draw each contour only for visualisation purposes
      cv2.drawContours(image, contours, i, (0, 0, 255), 2)
    # Find the orientation of each shape
      kq =  getOrientation(c, image)
      print("kq: ", kq)
  if checkitem[count] == 4.0:

    list_bb = myList[count]
    point = (list_bb[0],list_bb[1])
    print("point :",point)
    color = (0, 255, 0)
    # Kích thước của điểm
    thickness = -1  # Đặt -1 để vẽ một điểm đầy đủ
    # Vẽ điểm trên hình ảnh
    cv2.circle(image, (int(list_bb[0]),int(list_bb[1])), 3, (0, 0, 255), thickness) #chấm điểm vào ảnh ở tọa độ tâm của bounding box của lỗ tâm
    # in tọa độ của tâm
    text = f"({int(list_bb[0])}, {int(list_bb[1])})"
    cv2.putText(image, text, (int(list_bb[0])+30,int(list_bb[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  count+=1


# cv2.imwrite('output_image2.jpg', result)
cv2.imwrite('output_image_Finish.jpg', image)


#Kết luận rằng lấy tâm của bounding box của lỗ tâm chuẩn hơn cách lấy tâm từ các điểm masks của lỗ tâm 