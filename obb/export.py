from ultralytics import YOLO
model = YOLO('yolov8n-obb.pt')
model.export(format='ncnn')