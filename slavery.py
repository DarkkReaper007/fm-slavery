#install dependencies
#pip install ultralytics
from ultralytics import YOLO
model_path = "" # path to the weights file (.pt file)
model = YOLO(model_path)
results = model.predict(source='', imgsz=640, conf=0.25) # put the folder that contains all the images
for result in results:
    print("Labels:", result.names)
    print("Confidences:", result.boxes.conf)
