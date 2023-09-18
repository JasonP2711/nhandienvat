#https://docs.ultralytics.com/modes/predict/

from ultralytics import YOLO
from PIL import Image
import cv2



# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('/content/runs/detect/train2/weights/best.pt')  # load a custom model

# Define path to the image file
# source = '/content/drive/MyDrive/NhanDienvat/NDVat/train/images/train/anhvat_692_png.rf.089a731feb7e3f2dce5db14f23d458e4.jpg'

#picture
# results = model.predict(source, save=True, imgsz=640, conf=0.5)
results = model('/content/drive/MyDrive/NhanDienvat/anhvat_123.png',save=True)


# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

