import numpy as np
from ultralytics import YOLO
from ultralytics.engine.model import Results
from typing import List

from variables import YOLO_VERSION
from variables import MIN_CONF

class Detector:

    def __init__(self):
      self.model = YOLO(YOLO_VERSION)
    
    def _classes_to_index(self, class_names: List[str]) -> List[int]:
      idx_classes = []
      for idx, name in self.model.names.items():
        if name in class_names:
          idx_classes.append(idx)
      return idx_classes

    def get_detection_results(self, frame: np.ndarray) -> Results:
      idx_classes = self._classes_to_index(["person"])
      return self.model.predict(frame, classes = idx_classes, conf=MIN_CONF, verbose=False)
    
    def show_classes(self):
      print(self.model.names)
