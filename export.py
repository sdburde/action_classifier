from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/yolo11_classifier2/weights/best.pt")

# Export the model to ONNX format
path = model.export(format="engine", imgsz=320)  # return path to exported model