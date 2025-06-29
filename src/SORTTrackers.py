import torch
from typing import List

from variables import VIDEO_PATH_TEST
from variables import MAX_FRAME_LOST

from src.videoManager import VideoManager
from src.detector import Detector
from src.matcher import Matcher
from src.predictor import Predictor
from src.track import Track
from src.visualizer import Visualizer

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

  def _tracks_to_matrix_xyxy(self, tracks: List[Track], produce_id: bool = False) -> torch.Tensor:
    """
    Converts a list of tracks into a (N,4) or (N,5) tensor of boxes in [x1, y1, x2, y2] format.
    If produce_id is True, includes the track ID as first column.
    """
    box_list = []
    id_list = []

    for track in tracks:
      track_xywh = track.X_hat.T[:, :4]
      box_list.append(track_xywh)
      if produce_id:
        track_id = torch.tensor([[track.identifier]], dtype=torch.float32, device=track_xywh.device)
        id_list.append(track_id)

    if not box_list:
      return torch.empty((0, 5 if produce_id else 4))

    boxes = torch.cat(box_list, dim=0)
    boxes_xyxy = self._xywh_to_xyxy(boxes)
    if produce_id:
      ids = torch.cat(id_list, dim=0)
      return torch.cat((ids, boxes_xyxy), dim=1)
    else:
      return boxes_xyxy
  

  def run_tracking(self, video_path= VIDEO_PATH_TEST):

    videoManager = VideoManager(video_path)
    detector = Detector()
    matcher = Matcher()
    predictor = Predictor(videoManager.dt)
    visualizer = Visualizer(save_video=False, dt=videoManager.dt)
    n_frame = 0

    for frame in videoManager:
      #-Stat
      print("Analyzing frame number -> ", n_frame)
      print("Elements in traks -> ", len(self.all_tracks))
      n_frame += 1

      #-Detection
      results = detector.get_detection_results(frame)[0]
      detections_xyxy = results.boxes.xyxy
      detections_xywh = results.boxes.xywh

      #-Matching
      tracks_xyxy = self._tracks_to_matrix_xyxy(self.all_tracks)
      matching = matcher.hungarian_algorithm(tracks_xyxy, detections_xyxy)
      print("MATCHING ->\n", matching)

      #-Prediction
      for pair in matching["assignments"]:
        track_to_udpdate = self.all_tracks[pair[0]]
        new_measures = detections_xywh[pair[1], :].unsqueeze(1)
        predictor.prediction_step(track_to_udpdate)
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
      printing_matrix = self._tracks_to_matrix_xyxy(self.all_tracks, produce_id=True)
      visualizer.draw(frame, printing_matrix)

      
     
