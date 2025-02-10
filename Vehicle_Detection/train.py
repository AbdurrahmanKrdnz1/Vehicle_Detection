# We need to import the ultralytics library to use the YOLOv8 model
import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    
    # We will use the pretrained weights for this model
    model = YOLO("yolov8n.pt")
    
    # Now we can train our model
    result = model.train(data = r"C:\Users\abdur\Desktop\Vehicle_Detection\Vehicle_Detection_Traffic.v1i.yolov8\data.yaml",
                     epochs = 100,
                     batch = 4,
                     imgsz = 640,
                     lr0 = 0.001,
                     device = "cuda",
                     save = True)