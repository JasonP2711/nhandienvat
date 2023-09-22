from ultralytics import YOLO
import cv2


image = cv2.imread(r'.\NDVat\train\images\20230919_104716_jpg.rf.747a0be5e965e191970e6a6c41c3ae68.jpg')

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO(r'.\NDVat\runs\segment\train\weights\best.pt')  # load a custom model

# Predict with the model
results = model(r'.\NDVat\train\images\20230919_104716_jpg.rf.747a0be5e965e191970e6a6c41c3ae68.jpg', save = True)  # predict on an image
#picture
# results = model.predict(source, save=True, imgsz=640, conf=0.5)


# View results
myList = []
for r in results:
    print(r.boxes.xywh)  # print the Boxes object containing the detection bounding boxes
    myList.append(r.boxes.xywh)


print(myList)

#còn lỗi


# ///////////////////////////////////////////////////////////

from ultralytics import YOLO
import cv2

link = '/content/nhandienvat/trainCoTam/train/images/20230919_104710_jpg.rf.5649bd50e968333b67fa7da411f178ec.jpg'

image = cv2.imread(link)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('output_image_gray.jpg', gray_image)

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('/content/runs/segment/train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model("/content/output_image_gray.jpg", save = True)  # predict on an image


myList = []

mylistmask = []

checkitem = []
for r in results:
    mylistmask = r.masks.xy
    # print("mask: ",r.masks[0].xy[0])
    print("shape: ",r.masks.shape)
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes
    myList = r.boxes.xywh.tolist()
    print(r.boxes.xywh.tolist())
    checkitem = r.boxes.cls.tolist()
    # print(int(checkitem[2]))
print("myList: ",myList)
list = myList[1]
# print(list)



#tu do dai cua danh sach cac diem mask cua tung doi tuong(4),
#ta lap vao tung doi tuong va lay cac diem tron tung doi tuong 


count = 0

# while count < 


# print("list masks: ",mylistmask.tolist()[0])

point = (list[0],list[1])
print("point :",point)
color = (0, 255, 0)
# Kích thước của điểm
thickness = -1  # Đặt -1 để vẽ một điểm đầy đủ

# Vẽ điểm trên hình ảnh
cv2.circle(image, point, 3, (0, 0, 255), thickness)


count = 0
while count < len(mylistmask):
  if checkitem[count] == 0.0:
    print("count: ", count)
    print(mylistmask[count].tolist())
    listitem = mylistmask[count].tolist()
    count2 = 0
    totalx= 0
    totaly = 0
    x = 0
    y = 0
    
    while count2 < len(listitem):
      print(listitem[count2])
      coodinate = listitem[count2]
      totalx = totalx + coodinate[0]
      totaly = totaly + coodinate[1]
      # cv2.circle(image, (int(coodinate[0]),int(coodinate[1])), 3, color, thickness)
      cv2.drawMarker(image, (coodinate[0],coodinate[1]), color, markerType=cv2.MARKER_STAR, markerSize=2)
      count2 = count2 + 1
    x = totalx / len(listitem)
    y = totaly / len(listitem)
    cv2.drawMarker(image, (int(x),int(y)), color, markerType=cv2.MARKER_STAR, markerSize=2)
  count = count + 1


cv2.imwrite('output_image2.jpg', image)
