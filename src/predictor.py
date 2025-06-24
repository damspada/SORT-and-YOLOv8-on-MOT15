import torch
from src.track import Track

class Prediction:

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

    # def prediction_step(self, person: Track):
    #   person.X_hat = self.A @ person.X_hat
    #   person.P = self.A @ person.P @ self.A.T + self.Q
    
    # def estimated

















