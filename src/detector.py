from ultralytics import YOLO
import numpy as np

class Detector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt") # try with larger models o Fine-Tuning
    
    def get_bounding_box(self, frame: np.ndarray):
      idx_person = self.model.names.index("person")
      return self.model.predict(frame, classes = [idx_person], conf=0.25)
    
    def show_classes(self):
      print(self.model.names)
