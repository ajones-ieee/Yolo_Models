from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
from yolo_detection_models import model_data

for model_name, resolution in model_data:
    model = YOLO(model_name)

    # Export the model to NCNN format
    model.export(format="ncnn", imgsz=resolution)  # creates NCNN version of the model
    
print('Done!')