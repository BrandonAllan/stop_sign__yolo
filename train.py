from ultralytics import YOLO
import torch

device_to_use = 0 if torch.cuda.is_available() else 'cpu'
print(f"Training will run on: {device_to_use}")

# Load the model
model = YOLO("yolo26n.pt")

# Train the model (pass the device right in here!)
results = model.train(
    data='data.yaml', 
    epochs=100,        
    batch=16,
    device=device_to_use
) 
