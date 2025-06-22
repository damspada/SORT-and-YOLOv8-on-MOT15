import cv2
import glob
import os

class VideoManager:
  def __init__(self, video_path):
    self.load_frames(video_path)
    self.MAX_FRAME = len(self.frame_paths)
  
  def load_frames(self, video_path):
    frame_paths = glob.glob(os.path.join(video_path, "*.jpg"))
    frame_paths.sort()
    self.frame_paths = frame_paths
    self.current_frame_idx = 0

  def get_next_frame(self):
    if self.current_frame_idx >= self.MAX_FRAME:
        return None
    frame_path = self.frame_paths[self.current_frame_idx]
    frame = cv2.imread(frame_path)
    self.current_frame_idx += 1
    return frame




        