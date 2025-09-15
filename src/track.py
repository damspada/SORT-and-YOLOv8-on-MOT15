import torch
from variables import MAX_FRAME_LOST, MIN_HITS, P_POSITION_INITIAL, P_DIMENSION_INITIAL, P_VELOCITY_INITIAL
class Track:
  _id_counter = 0

  def __init__(self, X0: torch.Tensor):
    """
    X0 is of the form [x,y,w,h]
    """
    self.identifier = Track._id_counter
    Track._id_counter += 1

    self.detections_missed = 0
    self.hits = 1
    self.hits_streak = 1

    V0 = torch.tensor([[1,1]], dtype=torch.float32)
    self.X_hat = torch.cat((X0, V0), dim=1).T #[x,y,w,h,vx,vy], Shape -> (6,1)

    # Initial state covariance matrix P - represents uncertainty in initial state estimates
    self.P = torch.tensor([
      [P_POSITION_INITIAL, 0, 0, 0,  0,   0],
      [0, P_POSITION_INITIAL, 0, 0,  0,   0],
      [0, 0, P_DIMENSION_INITIAL, 0,  0,   0],
      [0, 0, 0, P_DIMENSION_INITIAL,  0,   0],
      [0, 0, 0, 0, P_VELOCITY_INITIAL,  0],
      [0, 0, 0, 0,  0,  P_VELOCITY_INITIAL]
    ], dtype=torch.float32)

  def update_hits(self):
    self.hits += 1
    self.hits_streak += 1

  def reset_detections_missed(self):
    self.detections_missed = 0
  
  def increase_detections_missed(self):
    self.detections_missed += 1
    self.hits_streak = 0
    return self.detections_missed >= MAX_FRAME_LOST

  def check_confirmed_track(self):
    return (self.hits >= MIN_HITS and self.detections_missed <= 1) or self.hits_streak >= MIN_HITS

  def extract_bbox_in_row(self):
    from src.predictor import Predictor
    return (Predictor.H @ self.X_hat).T
