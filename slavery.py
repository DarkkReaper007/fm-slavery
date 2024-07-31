#Install dependencies
#pip install ultralytics

from ultralytics import YOLO


model_path = ""  #path to the weights file
model = YOLO(model_path)


image_path = "" #path to the image
while True:
    results = model.predict(source=image_path, imgsz=640, conf=0.25)
    for result in results:
        if result.boxes:  # Check if there are any predictions
            print(result.names)
            print("Confidences:", result.boxes.conf.tolist())  
        







