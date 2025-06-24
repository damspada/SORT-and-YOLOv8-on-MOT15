import torch

class Track:
  _id_counter = 0
  def __init__(self, X0: torch.Tensor):
    
    self.id = _id_counter
    _id_counter += 1

    V0 = torch.tensor([[1,1]], dtype=torch.float32)
    self.X_hat = torch.cat((X0, V0), dim=1).T

    self.P = torch.tensor([
      [9, 0, 0, 0,  0,   0],
      [0, 9, 0, 0,  0,   0],
      [0, 0, 4, 0,  0,   0],
      [0, 0, 0, 4,  0,   0],
      [0, 0, 0, 0, 900,  0],
      [0, 0, 0, 0,  0,  900]
    ])