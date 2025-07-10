import cv2
import glob
import numpy as np
import os
import configparser
from variables import SEQ_INFO

class VideoManager:
  """
  This class manages the loading and iteration of video frames from a MOT sequence.
  Parses `seqinfo.ini` to extract metadata (e.g., frame rate, total frames).
  """
    
  def __init__(self, video_path):
    """
    self.file_name: str
    self.dt: float
    self._MAX_FRAME: int
    self._frame_paths: List[str]
    self._current_frame_idx: int
    """
    img_ext = self._load_seqinfo(video_path)
    self._load_frames(video_path, img_ext)
  
  def __iter__(self):
    # FOR MULTIPLE ITERATION EDIT THIS PART
    self._current_frame_idx = 0
    return self

  def __next__(self) -> np.ndarray:
    if self._current_frame_idx >= self._MAX_FRAME:
      raise StopIteration
    frame_path = self._frame_paths[self._current_frame_idx]
    frame = cv2.imread(frame_path)
    self._current_frame_idx += 1
    return frame

  def _load_seqinfo(self,video_path) -> str:
    seqinfo_path = os.path.join(video_path, SEQ_INFO)
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    frame_rate = int(config['Sequence']['frameRate'])
    self.dt = 1.0 / frame_rate
    self._MAX_FRAME = int(config['Sequence']['seqLength'])
    self.file_name = config['Sequence']['name']
    return config['Sequence']['imExt']

  def _load_frames(self, video_path, img_ext):
    frame_paths = glob.glob(os.path.join(video_path, "img1/*" + img_ext))
    frame_paths.sort()
    self._frame_paths = frame_paths
    self._current_frame_idx = 0




        