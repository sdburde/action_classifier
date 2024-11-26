import os

from ultralytics11 import YOLO


# model = YOLO('runs/classify/yolo11_classifier2/weights/best.engine')
model = YOLO('runs/classify/yolo11_classifier4/weights/best.pt')


for img in os.listdir('test'):
    results = model(f"test/{img}", save=True, conf=0.5)
    results[0].show()


# results = model(f"test/images.jpeg", save=True, conf=0.5)
# results[0].show()