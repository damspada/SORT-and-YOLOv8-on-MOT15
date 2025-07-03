import torch
from typing import List

from src.videoManager import VideoManager
from src.detector import Detector
from src.matcher import Matcher
from src.predictor import Predictor
from src.track import Track
from src.visualizer import Visualizer
from src.matrixUtils import MatrixUtils
from src.metrics import MetricType

class SORTTrackers:
  """
  Core class implementing the SORT algorithm.
  Maintains a list of active tracks, updates them using detections via
  Kalman filtering and matching (e.g., IoU-based), spawns new tracks for unmatched detections,
  and removes tracks that have not been updated for a specified number of frames.
  """

  def __init__(self):
    self.all_tracks = []  

  def run_tracking(self, input_path, output_path=None):
    videoManager = VideoManager(input_path)
    detector = Detector()
    matcher = Matcher()
    predictor = Predictor(videoManager.dt)
    visualizer = Visualizer(
      output_path = output_path,
      file_name = videoManager.file_name,
      dt = videoManager.dt
    )
    n_frame = 0

    for frame in videoManager:
      #-Stat
      print("Analyzing frame number -> ", n_frame)
      n_frame += 1

      #-Detection
      results = detector.get_detection_results(frame)[0]
      detections = results.boxes
      detections_xywh = detections.xywh

      #-Prediction before matching
      for track in self.all_tracks:
        predictor.prediction_step(self.all_tracks)

      #-Matching
      matching = matcher.hungarian_algorithm(self.all_tracks, detections, MetricType.IOU)

      #-Prediction after matching
      for pair in matching["assignments"]:
        track_to_udpdate = self.all_tracks[pair[0]]
        new_measures = detections_xywh[pair[1], :].unsqueeze(1)
        predictor.estimated_step(track_to_udpdate, new_measures)
      
      tracks_to_delete = []
      for index in matching["lost_tracks"]:
        track_to_predict = self.all_tracks[index]
        to_delect = track_to_predict.increse_detections_missed()
        if to_delect:
          tracks_to_delete.append(index)
        else:
          predictor.prediction_step(track_to_predict)
      
      # Delete tracks lost for too long
      for index in sorted(tracks_to_delete, reverse=True):
        del self.all_tracks[index]

      for index in matching["new_detections"]:
        X0 = detections_xywh[index:index+1]
        new_track = Track(X0)
        self.all_tracks.append(new_track)
      
      #-Visualizer
      printing_matrix = MatrixUtils.tracks_to_matrix_xyxy(self.all_tracks, produce_id=True)
      visualizer.draw(frame, printing_matrix)

      
     
