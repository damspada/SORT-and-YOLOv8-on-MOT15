import torch
from typing import List

class Matcher:
  """
  The matching is done with the Hungarian algorithm and the IoU metric
  """
  def __init__(self):
    pass
  
  def _IoU(A: torch.Tensor, B: torch.Tensor) -> float:
    """
    Shape of A and B is [x_sx, y_sx, x_dx, y_dx]
    """
    intersection_x = max(0, min(A[2], B[2] - max(A[0], B[0])))
    intersection_y = max(0, min(A[3], B[3] - max(A[1], B[1])))
    intersection_boxes = intersection_x * intersection_y
    if intersection_boxes:
      wA, wB = A[2]-A[0], B[2]-B[0]
      hA, hB = A[3]-A[1], B[3]-B[1]
      union_boxes = (wA * hA) + (wB * hB) - intersection_boxes
      return intersection_boxes/union_boxes
    else:
      return intersection_boxes
  
  def match_tracks_and_detections(tracks: List[torch.Tensor], detections: List[torch.Tensor]):






