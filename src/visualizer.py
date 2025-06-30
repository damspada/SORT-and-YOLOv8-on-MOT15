import cv2
import os
import numpy as np
import time
from typing import Optional
import torch
import sys  #COLAB

IN_COLAB = 'google.colab' in sys.modules  #COLAB

class Visualizer:
  def __init__(self, output_path: Optional[str], file_name: str, dt: float):
    self.output_path = output_path
    self.file_name = file_name
    self.dt = dt
    self.writer = None
    self.colors = {}
    self.initialized = False
    self.last_frame_time = time.time()

    if self.output_path is not None:
      os.makedirs(self.output_path, exist_ok=True)
      video_path = os.path.join(self.output_path, f"{self.file_name}.mp4")
      if os.path.exists(video_path):
        os.remove(video_path)
      self.video_path = video_path
    else:
      self.video_path = None

  def _get_color(self, obj_id: int):
    if obj_id not in self.colors:
      np.random.seed(obj_id)
      self.colors[obj_id] = tuple(np.random.randint(0, 255, size=3).tolist())
    return self.colors[obj_id]

  def _init_writer(self, frame_shape):
    height, width = frame_shape[:2]
    fps = 1.0 / self.dt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    self.writer = cv2.VideoWriter(self.video_path, fourcc, fps, (width, height))

  def draw(self, frame: np.ndarray, boxes: torch.Tensor):
    if self.video_path and not self.writer:
      self._init_writer(frame.shape)

    boxes_np = boxes.cpu().numpy()
    for box in boxes_np:
      obj_id, x1, y1, x2, y2 = box.astype(int)
      color = self._get_color(obj_id)
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
      cv2.putText(frame, str(obj_id), (x1, y1 - 10),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if not IN_COLAB:                        #COLAB
      cv2.imshow("SORT Tracking", frame)    #COLAB
      if cv2.waitKey(1) & 0xFF == ord('q'): #COLAB
        self.close()                        #COLAB

    if self.writer:
      self.writer.write(frame)

    elapsed = time.time() - self.last_frame_time
    wait_time = max(self.dt - elapsed, 0)
    if wait_time > 0:
      time.sleep(wait_time)
    self.last_frame_time = time.time()

  def close(self):
    if self.writer:
      self.writer.release()
    if not IN_COLAB:           #COLAB
      cv2.destroyAllWindows()  #COLAB



      
