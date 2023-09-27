
# ///////////////////////////////////////////////////////////

from ultralytics import YOLO
import cv2
import numpy as np

import argparse
from math import atan2, cos, sin, sqrt, pi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

link = r'E:\My_Code\NhanDienvat\detectByYolov8\dataset_specialItems\train\images\20230922_155254_jpg.rf.53417c933aae3540eb4f59ccc6e42b39.jpg'

image = cv2.imread(link)

def drawAxis(img, p_, q_, colour, scale,degree):
 p = list(p_)
 q = list(q_)
 print("diemP: ",p)
 print("diemQ: ",q)
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
 
 
 cv2.circle(img, cntr, 2, (255, 0, 255), 1)
 p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
 p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
#  cv.circle(img, (eigenvectors[0,1], eigenvectors[0,0]), 2, (0, 0, 255), 1)
 angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
 degree = angle * (180 / np.pi)

 drawAxis(img, cntr, p1, (0, 255, 0), 2,degree)
 drawAxis(img, cntr, p2, (255, 255, 0), 5,degree)
 return degree


beta = -50  # Giảm độ sáng

# Sử dụng hàm cv2.convertScaleAbs để giảm độ sáng của ảnh
src = cv2.convertScaleAbs(image, alpha=1, beta=beta)


# Check if image is loaded successfully
if src is None:
 print('Could not open or find the image: ', args.input)
 exit(0)
cv2.imshow('src', src)
# Convert image to grayscale
gray_picture = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

gray_inverted = cv2.bitwise_not(gray_picture)
# Convert image to binary
_, bw = cv2.threshold(gray_picture, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(contours):
 # Calculate the area of each contour
 area = cv2.contourArea(c)
 # Ignore contours that are too small or too large
 if area < 1e2*3 or 1e5 < area:
  continue
 # Draw each contour only for visualisation purposes
 cv2.drawContours(src, contours, i, (0, 0, 255), 2)
 # Find the orientation of each shape
 getOrientation(c, src)
 cv2.imwrite('orientation_image.jpg',src)
cv2.imshow('output', src)
# Assuming you have a grayscale image in the 'gray_picture' variable
# Normalize the grayscale values to be between 0 and 1

plt.imshow(bw, cmap="gray")
plt.title("Bitwise Gray")
plt.axis("off")
plt.show()

plt.imshow(gray_inverted, cmap="gray")
plt.title("gray_inverted Gray")
plt.axis("off")
plt.show()

# ///////////////////////////////////

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(r'E:\My_Code\NhanDienvat\runs\segment\train\weights\best.pt')  # load a custom model

# Predict with the model
results = model(r"E:\My_Code\NhanDienvat\detectByYolov8\dataset_specialItems\train\images\20230922_155058_jpg.rf.f9c8117af8571624c38fa89e5ba3b55c.jpg", save = True)  # predict on an image


myList = []

mylistmask = []

checkitem = []
for r in results:
    # mylistmask = r.masks.xy #ssử dụng nếu dùng phương pháp tìm tọa độ tâm theo các điểm masks
    # print("mask: ",r.masks[0].xy[0])
    # print("shape: ",r.masks.shape)
    print("masks: ",r.masks)
    print("boxes: ",r.boxes)  # print the Boxes object containing the detection bounding boxes
    myList = r.boxes.xywh.tolist()
    print(r.boxes.xywh.tolist())
    checkitem = r.boxes.cls.tolist()
    # print(int(checkitem[2]))
print("myList: ",myList)
print("length: ",len(myList))


# print(list)
count = 0

while count < len(checkitem):
  print("count: ",count)
  if checkitem[count] == 4.0:
    list = myList[count]
    point = (list[0],list[1])
    print("point :",point)
    color = (0, 255, 0)
    # Kích thước của điểm
    thickness = -1  # Đặt -1 để vẽ một điểm đầy đủ
    # Vẽ điểm trên hình ảnh
    cv2.circle(image, (int(list[0]),int(list[1])), 3, (0, 0, 255), thickness) #chấm điểm vào ảnh ở tọa độ tâm của bounding box của lỗ tâm
    # in tọa độ của tâm
    text = f"({int(list[0])}, {int(list[1])})"
    cv2.putText(image, text, (int(list[0])+30,int(list[1])+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.line(image, (int(list[0]),int(list[1])), (int(list[0]) + 50, int(list[1])), (0, 0, 255), 2)  # Vẽ trục X (màu đỏ)
    cv2.line(image, (int(list[0]),int(list[1])), (int(list[0]), int(list[1]) + 50), (0, 255, 0), 2)  # Vẽ trục Y (màu xanh lá)
  count = count + 1
# print("list masks: ",mylistmask.tolist()[0])



# ///////////////////Cách lấy tâm từ các điểm masks/////////////
# count = 0
# while count < len(mylistmask):
#   if checkitem[count] == 4.0:
#     print("count: ", count)
#     print(mylistmask[count].tolist())
#     listitem = mylistmask[count].tolist()
#     count2 = 0
#     totalx= 0
#     totaly = 0
#     x = 0
#     y = 0
    
#     while count2 < len(listitem):
#       print(listitem[count2])
#       coodinate = listitem[count2]
#       totalx = totalx + coodinate[0]
#       totaly = totaly + coodinate[1]
#       # cv2.circle(image, (int(coodinate[0]),int(coodinate[1])), 3, color, thickness)
#       cv2.drawMarker(image, (int(coodinate[0]),int(coodinate[1])), color, markerType=cv2.MARKER_STAR, markerSize=2)
#       count2 = count2 + 1
#     x = totalx / len(listitem)
#     y = totaly / len(listitem)
#     cv2.drawMarker(image, (int(x),int(y)), color, markerType=cv2.MARKER_STAR, markerSize=2)
#   count = count + 1


cv2.imwrite('output_image2.jpg', image)


#Kết luận rằng lấy tâm của bounding box của lỗ tâm chuẩn hơn cách lấy tâm từ các điểm masks của lỗ tâm 