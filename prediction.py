from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('best.pt')  # load a custom model

# Predict with the model
results = model('20230919_104710_jpg.rf.82ed36d6e0d9e6512348e114d1a0a375.jpg', save = True)  # predict on an image
#picture
# results = model.predict(source, save=True, imgsz=640, conf=0.5)



# View results
for r in results:
    print(r.boxes)  # print the Boxes object containing the detection bounding boxes

