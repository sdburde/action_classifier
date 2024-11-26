import os
import time  # To measure time

from ultralytics11 import YOLO

# Measure model load time
start_time = time.time()
model = YOLO('runs/classify/yolo11_classifier2/weights/best.pt')
load_time = time.time() - start_time
print(f"Model load time: {load_time:.4f} seconds")

# Loop through images and measure inference time
for img in os.listdir('cropped'):
    img_path = f"cropped/{img}"
    
    # Measure inference time
    start_time = time.time()
    results = model(img_path, save=True, conf=0.5)

    # print(results[0])
    # print(results[0].probs)
    print('\n\n\n\n')
    print(results[0].probs.top5)
    print(results[0].probs.top5conf)
    print('\n\n\n\n')

    inference_time = time.time() - start_time
    print(f"Inference time for {img}: {inference_time:.4f} seconds")
    
    # # Display results
    # results[0].show()
