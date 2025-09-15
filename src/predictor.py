import torch
from src.track import Track
from variables import Q_POSITION_NOISE, Q_DIMENSION_NOISE, Q_VELOCITY_NOISE

class Predictor:
  """
  The prediction is done with the Kalman Filter algorithm
  """
  
  # Process noise covariance matrix Q - defines uncertainty in motion model
  # It could change after test
  Q = torch.tensor([
    [Q_POSITION_NOISE, 0,  0,   0,   0,   0 ],
    [0, Q_POSITION_NOISE,  0,   0,   0,   0 ],
    [0, 0, Q_DIMENSION_NOISE,   0,   0,   0 ],
    [0, 0,  0,  Q_DIMENSION_NOISE,   0,   0 ],
    [0, 0,  0,   0,  Q_VELOCITY_NOISE,  0 ],
    [0, 0,  0,   0,   0,  Q_VELOCITY_NOISE]
  ], dtype=torch.float32)
  
  H = torch.tensor([
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0]
  ], dtype=torch.float32)

  # It could change after test
  R = torch.tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ], dtype=torch.float32)

  def __init__(self, dt):

    # acceleration could be inserted
    self.A = torch.tensor([
      [1, 0, 0, 0, dt,  0],
      [0, 1, 0, 0,  0, dt],
      [0, 0, 1, 0,  0,  0],
      [0, 0, 0, 1,  0,  0],
      [0, 0, 0, 0,  1,  0],
      [0, 0, 0, 0,  0,  1]
    ], dtype=torch.float32) 

  def prediction_step(self, track: Track):
    """
    Estimate the bounding box of a track based on information from the previous iteration.
    """
    track.X_hat = self.A @ track.X_hat
    track.P = self.A @ track.P @ self.A.T + Predictor.Q
  
  def estimated_step(self, track:Track, Z: torch.Tensor):
    """
    Use the data provided by the detector to accurately update the bounding box.
    """
    K = track.P @ Predictor.H.T @ torch.inverse(Predictor.H @ track.P @ Predictor.H.T + Predictor.R)
    track.X_hat = track.X_hat + K @ (Z - Predictor.H @ track.X_hat)
    track.P = track.P - K @ Predictor.H @ track.P


















