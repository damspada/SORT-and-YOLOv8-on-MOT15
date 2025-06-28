import torch
from variables import MAX_FRAME_LOST
class Track:
  _id_counter = 0

  def __init__(self, X0: torch.Tensor):
    """
    X0 is of the form [x,y,w,h]
    """
    self.id = Track._id_counter
    Track._id_counter += 1

    self.detections_missed = 0

    V0 = torch.tensor([[1,1]], dtype=torch.float32)
    self.X_hat = torch.cat((X0, V0), dim=1).T #[x,y,w,h,vx,vy]

    self.P = torch.tensor([
      [9, 0, 0, 0,  0,   0],
      [0, 9, 0, 0,  0,   0],
      [0, 0, 4, 0,  0,   0],
      [0, 0, 0, 4,  0,   0],
      [0, 0, 0, 0, 900,  0],
      [0, 0, 0, 0,  0,  900]
    ], dtype=torch.float32)

  def increse_detections_missed(self):
    self.detections_missed += 1
    return self.detections_missed >= MAX_FRAME_LOST



