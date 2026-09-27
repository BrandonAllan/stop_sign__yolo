from ultralytics import YOLO
import os

model = YOLO("runs/detect/train/weights/best.pt")

dataset_path = "stop_sign_dataset" 

results = model.predict(source=dataset_path, save=True, conf=0.8)

print("\n--- center pixel coordinates---")
for result in results:
    file_name = os.path.basename(result.path)
    boxes = result.boxes
    
    if len(boxes) == 0:
        print(f"[{file_name}]: No stop sign detected.")
        continue
        
    for box in boxes:
        x_center, y_center, w, h = box.xywh[0]
        print(f"[{file_name}]: Center (X: {int(x_center)}, Y: {int(y_center)})")