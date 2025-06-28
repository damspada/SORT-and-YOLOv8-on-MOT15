import torch
from src.track import Track

class Predictor:
  """
  The prediction is done with the Kalman Filter algorithm
  """

  def __init__(self, dt):

    # acceleration could be inserted
    self.A = torch.tensor([
      [1, 0, 0, 0, dt, 0],
      [0, 1, 0, 0, 0,  dt],
      [0, 0, 1, 0, 0,  0],
      [0, 0, 0, 1, 0,  0],
      [0, 0, 0, 0, 1,  0],
      [0, 0, 0, 0, 0,  1]
    ], dtype=torch.float32)

    # It could change after test
    self.Q = torch.tensor([
      [4, 0,  0,   0,   0,   0],
      [0, 4,  0,   0,   0,   0],
      [0, 0, 100,  0,   0,   0],
      [0, 0,  0,  100,  0,   0],
      [0, 0,  0,   0,  100,  0],
      [0, 0,  0,   0,   0,  100]
    ], dtype=torch.float32)

    self.H = torch.tensor([
      [1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0, 0]
    ], dtype=torch.float32)

    # It could change after test
    self.R = torch.tensor([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ], dtype=torch.float32) 

  def prediction_step(self, track: Track):
    """
    Estimate the bounding box of a track based on information from the previous iteration.
    """
    track.X_hat = self.A @ track.X_hat
    track.P = self.A @ track.P @ self.A.T + self.Q
    # return track.X_hat
  
  def estimated_step(self, track:Track, Z: torch.Tensor):
    """
    Use the data provided by the detector to accurately update the bounding box.
    """
    K = track.P @ self.H.T @ torch.inverse(self.H @ track.P @ self.H.T + self.R)
    track.X_hat = track.X_hat + K @ (Z - self.H @ track.X_hat)
    track.P = track.P - K @ self.H @ track.P
    # return track.X_hat


















