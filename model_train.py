import numpy as np
import cv2


from ultralytics import YOLO


model = YOLO("yolov8n.yaml")  # building a new model from YAML


# # Training the model
results = model.train(data="data.yaml", epochs=13, resume=True, batch=8, imgsz=800)


