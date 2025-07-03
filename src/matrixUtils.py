import torch
from typing import List, Tuple
from src.track import Track

class MatrixUtils:

  @staticmethod
  def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,4) [x1,y1,x2,y2] in a matrix (N,4) [cx,cy,w,h]
    """
    WH = boxes[:, 2:] - boxes[:, :2]
    XY_c = boxes[:, :2] + WH / 2
    return torch.cat((XY_c, WH), dim=1) 


  @staticmethod
  def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,4) [cx,cy,w,h] in a matrix (N,4) [x1,y1,x2,y2]
    """
    XY_1 = boxes[:, :2] - (boxes[:, 2:] / 2)
    XY_2 = boxes[:, :2] + (boxes[:, 2:] / 2)
    return torch.cat((XY_1, XY_2), dim=1)
  

  @staticmethod
  def rectangle_to_square(matrix: torch.Tensor) -> torch.Tensor:
    """
    Trasform a matrix (N,M) in:
      (N,N) if N > M
      (M,M) if M > N
    All new entries are set to 1e6 to discourage assignments by the algorithm.
    """
    N, M = matrix.shape[0], matrix.shape[1]
    if N > M:
      matrix = torch.cat([matrix, torch.full((N, N-M), 1e6)], dim=1)
    if M > N:
      matrix = torch.cat([matrix, torch.full((M-N, M), 1e6)], dim=0)
    return matrix


  @staticmethod
  def tracks_to_matrix_xyxy(tracks: List[Track], produce_id: bool = False) -> torch.Tensor:
    """
    Converts a list of tracks into a (N,4) or (N,5) tensor of boxes in [x1, y1, x2, y2] format.
    If produce_id is True, includes the track ID as first column.
    """
    box_list = []
    id_list = []

    for track in tracks:
      track_xywh = track.extract_bbox_in_row()
      box_list.append(track_xywh)
      if produce_id:
        track_id = torch.tensor([[track.identifier]], dtype=torch.float32, device=track_xywh.device)
        id_list.append(track_id)

    if not box_list:
      return torch.empty((0, 5 if produce_id else 4))

    boxes = torch.cat(box_list, dim=0)
    boxes_xyxy = MatrixUtils.xywh_to_xyxy(boxes)
    if produce_id:
      ids = torch.cat(id_list, dim=0)
      return torch.cat((ids, boxes_xyxy), dim=1)
    else:
      return boxes_xyxy

  @staticmethod
  def tracks_to_matrix_xywh_and_P(tracks: List[Track], produce_id: bool = False) -> Tuple(torch.Tensor, torch.Tensor):
    """
    Converts a list of tracks into a (N,4) tensor of boxes in [x, y, w, h] format.
    And returns a tensor (N,6,6) with all the matrices P of each track.
    """
    box_list = []
    P_list = []

    for track in tracks:
        track_xywh = track.extract_bbox_in_row()
        box_list.append(track_xywh)
        P_list = track.P

    boxes_xywh = torch.cat(box_list, dim=0)
    tensor_P = 0
    return boxes_xywh, tensor_P





