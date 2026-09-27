# YOLOv26 Stop Sign Detector

A deep learning computer vision project for detecting STOP traffic signs in images using a custom-trained YOLO26n object detection model. The project performs object detection, bounding-box prediction, confidence scoring, and pixel-coordinate localization of detected STOP signs.

## How It Works

The detection pipeline follows these steps:

1. **Model Loading:** The script loads the custom pre-trained YOLOv8 weights (`best.pt`).
2. **Image Inference:** Input images are passed through the object-detection model.
3. **Bounding Box & Class Filtering:** Bounding-box coordinates and class predictions are extracted for each detected STOP sign. Predictions below the selected confidence threshold are discarded.
4. **Center Position Calculation:** The center coordinates (X, Y) of each bounding box are calculated in pixel coordinates.
5. **Terminal Output:** Image filenames and corresponding center pixel coordinates are printed to the console.
6. **Output Visualization:** Annotated images with predicted bounding boxes, labels, and confidence scores are automatically saved.

## Project Structure

```text
.
├── data.yaml
├── train.py
├── test.py
├── runs/
│   └── detect/
│       ├── train/
│       │   └── weights/
│       │       └── best.pt
│       └── predict/
│           └── annotated images
└── README.md
```

## Dataset

The model was trained using a STOP-sign object-detection dataset containing annotated images.

The dataset is not included directly in this repository. It can be obtained from the original dataset source and used to reproduce the training process.

Dataset source:

`https://universe.roboflow.com/sign-detection-h24ey/stop-sign-h0vwm/dataset/2`



