
# ///////////////////////////////////////////////////////////

from ultralytics import YOLO
import cv2

link = r'E:\My_Code\NhanDienvat\detectByYolov8\dataset_specialItems\train\images\20230922_155058_jpg.rf.f9c8117af8571624c38fa89e5ba3b55c.jpg'

image = cv2.imread(link)

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('output_image_gray.jpg', gray_image)

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