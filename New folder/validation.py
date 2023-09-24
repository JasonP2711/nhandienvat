from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('/content/runs/segment/train/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered
metrics.box.map    # map50-95(B)
metrics.box.map50  # map50(B)
metrics.box.map75  # map75(B)
metrics.box.maps   # a list contains map50-95(B) of each category
metrics.seg.map    # map50-95(M)
metrics.seg.map50  # map50(M)
metrics.seg.map75  # map75(M)
metrics.seg.maps   # a list contains map50-95(M) of each category
# /////////////////////////////////////////////////////


from ultralytics import YOLO
import cv2


image = cv2.imread('/content/nhandienvat/anhvat_123.png')

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('/content/runs/segment/train3/weights/best.pt')  # load a custom model

# Predict with the model
results = model('/content/nhandienvat/anhvat_123.png', save = True)  # predict on an image


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

point = (int(list[0]),int(list[1]))
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
      cv2.drawMarker(image, (int(coodinate[0]),int(coodinate[1])), color, markerType=cv2.MARKER_STAR, markerSize=2)
      count2 = count2 + 1
    x = totalx / len(listitem)
    y = totaly / len(listitem)
    cv2.drawMarker(image, (int(x),int(y)), color, markerType=cv2.MARKER_STAR, markerSize=2)
  count = count + 1


cv2.imwrite('output_image.jpg', image)
