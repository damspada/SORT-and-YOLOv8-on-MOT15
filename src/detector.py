from ultralytics import YOLO
import numpy as np

class Detector:

    def __init__(self):
        self.yolo = YOLO("yolov8n.pt") # try with larger models o Fine-Tuning
    
    def get_bounding_box(self, frame: np.ndarray):
      return self.yolo(frame, conf=0.25)