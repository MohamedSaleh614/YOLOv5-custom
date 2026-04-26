# YOLOv5 Custom Implementation

A custom YOLOv5 model implemented from scratch in PyTorch for object detection.

## Project Files

- `nn.py`          → YOLOv5 model architecture (ConvBNSILU, Bottleneck, C3, SPPF, PANet)
- `dataset.py`     → Custom dataset loader with augmentation and target building
- `loss.py`        → CIoU + Objectness + Classification loss
- `train.py`       → Training script with mixed precision and Adam optimizer
- `test.py`        → Inference script with Non-Max Suppression and OpenCV visualization

## Features

- Built from scratch (no Ultralytics dependency)
- Supports single class detection (easy to extend to multiple classes)
- Uses C3 blocks, SPPF, and Feature Pyramid Network
- Complete training and inference pipeline
