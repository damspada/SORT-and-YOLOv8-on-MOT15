import torch
from typing import List

from variables import VIDEO_PATH_TEST
from variables import MAX_FRAME_LOST

from videoManager import VideoManager
from detector import Detector
from matcher import Matcher
from predictor import Predictor
from track import Track
# from visualizer import Visualizer

class SORTTrackers:
  """
  Core class implementing the SORT algorithm.
  Maintains a list of active tracks, updates them using detections via
  Kalman filtering and matching (e.g., IoU-based), spawns new tracks for unmatched detections,
  and removes tracks that have not been updated for a specified number of frames.
  """

  def __init__(self):
    self.all_tracks = []

  # @staticmethod
  # def _xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
  #   """
  #   Trasform a matrix (N,4) [x1,y1,x2,y2] in a matrix (N,4) [cx,cy,w,h]
  #   """
  #   WH = boxes[:, 2:] - boxes[:, :2]
  #   XY_c = boxes[:, :2] + WH / 2
  #   return torch.cat((XY_c, WH), dim=1) 

  @staticmethod
  def _xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,4) [cx,cy,w,h] in a matrix (N,4) [x1,y1,x2,y2]
    """
    XY_1 = boxes[:, :2] - (boxes[:, 2:] / 2)
    XY_2 = boxes[:, :2] + (boxes[:, 2:] / 2)
    return torch.cat((XY_1, XY_2), dim=1) 

  def _tracks_to_matrix_xyxy(self, tracks: List[Track]) -> torch.Tensor:
    """
    Produces a matrix (N,4) with the boxes of all tracks in the form [x,y,x,y]
    """
    track_matrix = torch.empty((0,4))
    for track in tracks:
      track_xywh = track.X_hat[:4]
      track_matrix = torch.cat((track_matrix, track_xywh), dim=1)

    return self._xywh_to_xyxy(track_matrix)

  def run_tracking(self, video_path= VIDEO_PATH_TEST):

    videoManager = VideoManager(video_path)
    detector = Detector()
    matcher = Matcher()
    predictor = Predictor(videoManager.dt)
    # visualizer = Visualizer()

    for frame in videoManager:
      #-Detection
      results = detector.get_detection_results(frame)
      detections_xyxy = results.boxes.xyxy
      detections_xywh = results.boxes.xywh

      #-Matching
      tracks_xyxy = self._tracks_to_matrix_xyxy(self.all_tracks)
      matching = matcher.hungarian_algorithm(tracks_xyxy, detections_xyxy)
      
      #-Prediction
      for pair in matching["assignments"]:
        track_to_udpdate = self.all_tracks[pair[0]]
        new_measures = detections_xywh[pair[1], :]
        predictor.prediction_step(track_to_udpdate)
        predictor.estimated_step(track_to_udpdate, new_measures)
      
      tracks_to_delete = []
      for index in matching["lost_tracks"]:
        track_to_predict = self.all_tracks[index]
        to_delect = track_to_predict.increse_detections_missed()
        if to_delect:
          tracks_to_delete.append[index]
        else:
          predictor.prediction_step(track_to_predict)
      
      # Delete tracks lost for too long
      for index in sorted(tracks_to_delete, reverse=True):
        del self.all_tracks[index]

      for index in matching["new_detections"]:
        new_track = Track(detections_xywh[index, :])
        self.all_tracks.append(new_track)
      
      #-Visualizer

      
     








