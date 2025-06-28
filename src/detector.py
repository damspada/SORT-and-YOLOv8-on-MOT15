from ultralytics import YOLO
import numpy as np
from ultralytics.engine.model import Results
from variables import YOLO_VERSION
from variables import MIN_CONF

class Detector:

    def __init__(self):
        self.model = YOLO(YOLO_VERSION) # try with larger models o Fine-Tuning
    
    # Possible to return a boxs [x,y,w,h]
    def get_detection_results(self, frame: np.ndarray) -> Results:
      idx_person = self.model.names.index("person")
      return self.model.predict(frame, classes = [idx_person], conf=MIN_CONF)
    
    def show_classes(self):
      print(self.model.names)
