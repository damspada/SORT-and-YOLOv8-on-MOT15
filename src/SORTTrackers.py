from variables import VIDEO_PATH_TEST
from variables import MAX_FRAME_LOST

from videoManager import VideoManager
from detector import Detector
from matcher import Matcher
from predictor import Predictor
# from visualizer import Visualizer

class SORTTrackers:
  """
  Core class implementing the SORT algorithm.
  Maintains a list of active tracks, updates them using detections via
  Kalman filtering and matching (e.g., IoU-based), spawns new tracks for unmatched detections,
  and removes tracks that have not been updated for a specified number of frames.
  """

  def __init__(self):
    self.all_track = []

  def run_tracking(video_path= VIDEO_PATH_TEST):

    videoManager = VideoManager(video_path)
    detector = Detector()
    matcher = Matcher()
    predictor = Predictor(videoManager.dt)
    # visualizer = Visualizer()

    for frame in videoManager:
      results = detector.get_detection_results(frame)
      detections_xyxy = results.boxes.xyxy

      
     








