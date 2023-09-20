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